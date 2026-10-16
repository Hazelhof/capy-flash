#include "artifact/materializer.h"

#include "artifact/framing.h"
#include "artifact/reader.h"
#include "core/startup.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <new>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cmath>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace ninfer::artifact {
namespace {

// T156: the host-mapped weights are one page-locked allocation (64.6 GiB for Qwen3.8-Flash-Next).
// Without enough free RAM the populate + registration stalls in reclaim and systemd-oomd kills the
// server or a neighbour (measured: 35 GiB held by another process -> memory pressure 36 % while
// loading). Refuse up front, with the numbers, unless NINFER_RAM_GUARD=0.
void require_host_memory(std::uint64_t bytes) {
#if defined(__linux__)
    if (const char* guard = std::getenv("NINFER_RAM_GUARD"); guard != nullptr && guard[0] == '0') {
        return;
    }
    std::ifstream meminfo("/proc/meminfo");
    std::uint64_t total_kib = 0, available_kib = 0;
    for (std::string line; std::getline(meminfo, line);) {
        std::istringstream fields(line);
        std::string key;
        std::uint64_t value = 0;
        fields >> key >> value;
        if (key == "MemTotal:") { total_kib = value; }
        if (key == "MemAvailable:") { available_kib = value; }
    }
    if (total_kib == 0 || available_kib == 0) { return; } // unreadable: no guard rather than a false refusal
    constexpr std::uint64_t kGiB = 1ULL << 30;
    const std::uint64_t total     = total_kib * 1024;
    const std::uint64_t available = available_kib * 1024;
    const std::uint64_t reserve   = std::max<std::uint64_t>(2 * kGiB, total / 20);
    if (available >= bytes + reserve) { return; }
    const auto gib = [](std::uint64_t value) {
        std::ostringstream text;
        text.setf(std::ios::fixed);
        text.precision(1);
        text << static_cast<double>(value) / static_cast<double>(1ULL << 30) << " GiB";
        return text.str();
    };
    throw ArtifactError("not enough free RAM for the host-mapped weights: they need " + gib(bytes) +
                        " of page-locked memory plus a " + gib(reserve) + " reserve, and " +
                        gib(available) + " is available (MemAvailable). Free memory (another model "
                        "still loaded?) or set NINFER_RAM_GUARD=0 to start anyway");
#else
    (void)bytes;
#endif
}

constexpr std::size_t kSlotBytes        = 64ULL * 1024 * 1024;
constexpr std::size_t kMaximumSlotCount = 4;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw ArtifactError(std::string(operation) + ": " + cudaGetErrorName(status) + ": " +
                            cudaGetErrorString(status));
    }
}

// Selects each upload device in turn and restores the caller's device. A single-device upload
// leaves the current device alone, as it always has.
class DeviceSelection {
public:
    explicit DeviceSelection(std::span<DeviceContext* const> devices) : devices_(devices) {
        if (devices_.size() > 1) {
            check_cuda(cudaGetDevice(&caller_), "query current device");
            active_ = true;
        }
    }

    ~DeviceSelection() {
        if (active_) { (void)cudaSetDevice(caller_); }
    }

    DeviceSelection(const DeviceSelection&)            = delete;
    DeviceSelection& operator=(const DeviceSelection&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return devices_.size(); }

    void select(std::size_t index) const {
        if (active_) {
            check_cuda(cudaSetDevice(devices_[index]->device), "select weight upload device");
        }
    }

private:
    std::span<DeviceContext* const> devices_;
    int caller_  = 0;
    bool active_ = false;
};

// A slot is refilled only after every device that read it has finished its copies. Each
// device's completion event is created on that device. Direct reads need a kPayloadAlignment
// destination, which a small page-locked allocation does not always get (the CUDA runtime may
// place it inside an earlier small pinned allocation's pages), so the slot aligns its own start.
class Slot {
public:
    Slot(std::size_t bytes, const DeviceSelection& selection)
        : buffer(bytes + kPayloadAlignment),
          data(static_cast<std::byte*>(buffer.data()) +
               (kPayloadAlignment - reinterpret_cast<std::uintptr_t>(buffer.data()) %
                                        kPayloadAlignment) %
                   kPayloadAlignment) {
        try {
            for (std::size_t i = 0; i < selection.size(); ++i) {
                selection.select(i);
                check_cuda(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming),
                           "create weight staging completion event");
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~Slot() { release(); }

    Slot(const Slot&)            = delete;
    Slot& operator=(const Slot&) = delete;

    void wait() {
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (pending[i]) {
                check_cuda(cudaEventSynchronize(events[i]), "wait for weight staging transfer");
                pending[i] = false;
            }
        }
    }

    PinnedHostBuffer buffer;
    std::byte* data = nullptr; // kPayloadAlignment-aligned start inside `buffer`.
    std::array<cudaEvent_t, kMaximumDevices> events{};
    std::array<bool, kMaximumDevices> pending{};

private:
    void release() noexcept {
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (pending[i]) { (void)cudaEventSynchronize(events[i]); }
            if (events[i]) { (void)cudaEventDestroy(events[i]); }
            events[i]  = nullptr;
            pending[i] = false;
        }
    }
};

// Also covers failure between a queued copy and its event record. Destruct before slots.
struct TransferCompletion {
    std::span<DeviceContext* const> devices;
    bool pending = true;

    ~TransferCompletion() {
        if (pending) {
            for (auto* device : devices) { (void)cudaStreamSynchronize(device->transfer_stream); }
        }
    }

    void finish() {
        for (auto* device : devices) {
            check_cuda(cudaStreamSynchronize(device->transfer_stream), "complete weight upload");
        }
        pending = false;
    }
};

struct CopyRange {
    std::size_t file       = 0;
    std::uint64_t begin    = 0;
    std::uint64_t end      = 0;
    std::byte* destination = nullptr;
    std::size_t device     = 0;
};

struct ReadSpan {
    std::size_t file    = 0;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

float read_divisor(const Reader& reader, ObjectHandle handle, const WeightGeometry& geometry,
                   std::span<const std::byte> host, MaterializationStats& stats) {
    if (geometry.format != QType::NVFP4) { return 0.0F; }
    // A stacked plane keeps one word per source; every one is checked, none is the parent's.
    const auto count = static_cast<std::size_t>(geometry.divisor_count);
    std::vector<std::byte> words(4 * count);
    if (!host.empty()) {
        std::copy_n(host.data() + geometry.divisor_offset, words.size(), words.data());
    } else {
        const auto& object = reader.directory().tensor(handle);
        reader.read_into(checked_add(object.offset, geometry.divisor_offset, "weight divisor"),
                         words);
        stats.read_bytes = checked_add(stats.read_bytes, words.size(), "read bytes");
    }
    float first = 0.0F;
    for (std::size_t i = 0; i < count; ++i) {
        const auto value = std::bit_cast<float>(read_u32_le(words.data() + 4 * i));
        if (!std::isfinite(value) || value <= 0) {
            throw ArtifactError(reader.directory().tensor(handle).id +
                                ": invalid NVFP4 weight divisor");
        }
        if (i == 0) { first = value; }
    }
    return count == 1 ? first : 0.0F;
}

// Reads objects straight into host memory (the host-mapped Flash-Next expert banks, 65 GiB in
// ~100 objects, read as one batch so the streams never wait for each other) with direct reads:
// the page-locked copy is the only one kept, so the page cache would only add a copy and evict
// everything else. The objects are not 4 KiB-aligned in the files, so each 64 MiB piece is read
// (O_DIRECT) into its stream's aligned bounce buffer covering the piece and copied out; two
// streams keep an NVMe busy (T173, 60 GiB: 13.2 s with 2, 18.4 s with 4, 38.4 s with 8).
// `progress(bytes read so far)` runs on the calling thread about every 250 ms.
struct HostRead {
    std::uint64_t offset; // logical payload offset
    std::span<std::byte> destination;
};

void parallel_read(const Reader& reader, std::span<const HostRead> reads,
                   const std::function<void(std::uint64_t)>& progress) {
    constexpr std::uint64_t kChunk = 64ULL << 20;
    constexpr std::size_t kStreams = 2;
    constexpr std::uint64_t kAlign = kPayloadAlignment;

    struct Piece {
        std::size_t file;
        std::uint64_t file_offset;
        std::span<std::byte> destination;
    };

    std::vector<Piece> pieces;
    for (const auto& [offset, destination] : reads) {
        for (const auto& segment : reader.segments(offset, destination.size())) {
            const auto to =
                destination.subspan(static_cast<std::size_t>(segment.destination_offset),
                                    static_cast<std::size_t>(segment.bytes));
            for (std::uint64_t at = 0; at < segment.bytes; at += kChunk) {
                pieces.push_back(
                    {segment.file_index, segment.file_offset + at,
                     to.subspan(static_cast<std::size_t>(at),
                                static_cast<std::size_t>(std::min(kChunk, segment.bytes - at)))});
            }
        }
    }
    std::atomic<std::uint64_t> read_bytes{0};

    // One bounce buffer per stream: the aligned range covering a piece is at most kChunk plus
    // one alignment unit at each end.
    struct Bounce {
        std::unique_ptr<std::byte, decltype(&std::free)> data{nullptr, &std::free};

        Bounce() {
            data.reset(static_cast<std::byte*>(std::aligned_alloc(kAlign, kChunk + 2 * kAlign)));
            if (!data) { throw std::bad_alloc(); }
        }
    };

    const auto read_piece = [&](const Piece& piece, Bounce& bounce) {
        const std::uint64_t begin  = piece.file_offset / kAlign * kAlign;
        const std::uint64_t head   = piece.file_offset - begin;
        const std::uint64_t needed = head + piece.destination.size();
        const std::uint64_t length = (needed + kAlign - 1) / kAlign * kAlign;
        std::uint64_t got          = 0;
        while (got < needed) {
            const auto received = reader.read_direct(
                piece.file, begin + got,
                {bounce.data.get() + got, static_cast<std::size_t>(length - got)});
            if (received == 0) { throw ArtifactError("direct read ended before the object"); }
            got += received;
        }
        std::memcpy(piece.destination.data(), bounce.data.get() + head, piece.destination.size());
        read_bytes.fetch_add(piece.destination.size(), std::memory_order_relaxed);
    };
    // The first read of each file opens its direct descriptor; do it before the streams share it.
    Bounce first;
    std::vector<bool> opened(reader.directory().files.size(), false);
    std::vector<const Piece*> rest;
    for (const auto& piece : pieces) {
        if (!opened[piece.file]) {
            opened[piece.file] = true;
            read_piece(piece, first);
        } else {
            rest.push_back(&piece);
        }
    }
    std::vector<std::future<void>> tasks;
    for (std::size_t stream = 0; stream < std::min(kStreams, rest.size()); ++stream) {
        tasks.push_back(std::async(std::launch::async, [&, stream] {
            Bounce bounce;
            for (std::size_t i = stream; i < rest.size(); i += kStreams) {
                read_piece(*rest[i], bounce);
            }
        }));
    }
    // Every task finishes before an error leaves this frame: they write into `destination`.
    std::exception_ptr failure;
    for (auto& task : tasks) {
        while (task.wait_for(std::chrono::milliseconds(250)) != std::future_status::ready) {
            progress(read_bytes.load(std::memory_order_relaxed));
        }
        try {
            task.get();
        } catch (...) {
            if (!failure) { failure = std::current_exception(); }
        }
    }
    if (failure) { std::rethrow_exception(failure); }
}

} // namespace

// Read-only shared mapping of one file range; the page cache owns residency.
class FileMapping {
public:
    FileMapping(const std::filesystem::path& path, std::uint64_t offset, std::uint64_t bytes) {
        const auto page    = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
        const auto aligned = offset / page * page;
        length_            = static_cast<std::size_t>(offset - aligned + bytes);
        const int fd       = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { throw ArtifactError(path.string() + ": cannot open for mapping"); }
        base_ = ::mmap(nullptr, length_, PROT_READ, MAP_SHARED, fd, static_cast<off_t>(aligned));
        ::close(fd);
        if (base_ == MAP_FAILED) {
            base_ = nullptr;
            throw ArtifactError(path.string() + ": mmap failed");
        }
        // Lookups touch scattered rows; read-ahead would only evict other pages.
        (void)::madvise(base_, length_, MADV_RANDOM);
        data_ = static_cast<const std::byte*>(base_) + (offset - aligned);
    }

    ~FileMapping() {
        if (base_ != nullptr) { ::munmap(base_, length_); }
    }

    FileMapping(const FileMapping&)            = delete;
    FileMapping& operator=(const FileMapping&) = delete;

    [[nodiscard]] const std::byte* data() const noexcept { return data_; }

private:
    void* base_            = nullptr;
    std::size_t length_    = 0;
    const std::byte* data_ = nullptr;
};

MaterializedArtifact::MaterializedArtifact()                                           = default;
MaterializedArtifact::~MaterializedArtifact()                                          = default;
MaterializedArtifact::MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
MaterializedArtifact& MaterializedArtifact::operator=(MaterializedArtifact&&) noexcept = default;

std::span<const MappedObjectSegment>
MaterializedArtifact::file_segments(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].file.empty()) {
        throw ArtifactError("object has no file mapping");
    }
    return objects_[handle.index].file;
}

const WeightGeometry& MaterializedArtifact::file_geometry(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || !objects_[handle.index].file_geometry) {
        throw ArtifactError("object has no mapped tensor geometry");
    }
    return *objects_[handle.index].file_geometry;
}

const WeightParent& MaterializedArtifact::device_parent(ObjectHandle handle, int device) const {
    if (!has_device(handle, device)) { throw ArtifactError("object has no device weight backing"); }
    return *objects_[handle.index].device[static_cast<std::size_t>(device)].parent;
}

const DeviceShard& MaterializedArtifact::device_shard(ObjectHandle handle, int device) const {
    if (!has_device(handle, device)) { throw ArtifactError("object has no device weight backing"); }
    return objects_[handle.index].device[static_cast<std::size_t>(device)].shard;
}

const WeightParent& MaterializedArtifact::host_parent(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || !objects_[handle.index].host) {
        throw ArtifactError("object has no Host weight backing");
    }
    return *objects_[handle.index].host;
}

std::span<const std::byte> MaterializedArtifact::host_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].host_data.empty()) {
        throw ArtifactError("object has no retained Host bytes");
    }
    return objects_[handle.index].host_data;
}

bool MaterializedArtifact::has_device(ObjectHandle handle, int device) const noexcept {
    return device >= 0 && device < stats_.device_count && handle.index < objects_.size() &&
           objects_[handle.index].device[static_cast<std::size_t>(device)].parent.has_value();
}

MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                 std::span<DeviceContext* const> devices,
                                 const StartupObserver* startup_observer) {
    if (plan.source != &reader || plan.object_count != reader.directory().objects.size()) {
        throw ArtifactError("materialization plan belongs to another load session");
    }
    if (plan.device_count < 1 || plan.device_count > static_cast<int>(kMaximumDevices) ||
        devices.size() != static_cast<std::size_t>(plan.device_count)) {
        throw ArtifactError("materialization devices differ from the plan device count");
    }
    const StartupObserver no_observer;
    const auto& observer = startup_observer ? *startup_observer : no_observer;
    std::uint64_t total  = 0;
    for (const auto& placement : plan.device_objects) {
        if (placement.copies.empty()) {
            total = checked_add(total, placement.bytes, "device payload bytes");
        }
        for (const auto& copy : placement.copies) {
            total = checked_add(total, copy.bytes, "device payload bytes");
        }
    }
    // Progress counts the host-mapped bytes too (64.6 GiB for Flash-Next, read before the uploads):
    // without them the log showed "1.2 % ... ETA 1h" for a 40-second load.
    std::uint64_t host_total = 0;
    for (const auto& placement : plan.host_mapped_objects) {
        host_total = checked_add(host_total, reader.directory().tensor(placement.object).bytes,
                                 "host-mapped payload bytes");
    }
    const std::uint64_t all_total = checked_add(total, host_total, "weight payload bytes");
    StartupPhaseScope phase(observer, StartupPhase::WeightsMaterialize, StartupProgressUnit::Bytes,
                            all_total);
    DeviceSelection selection(devices);
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    out.stats_.device_count          = plan.device_count;
    out.stats_.file_bytes            = reader.file_bytes();
    out.stats_.read_bytes            = plan.prior_read_bytes;
    out.stats_.owned_value_bytes     = plan.owned_value_bytes;
    out.stats_.device_capacity_bytes = plan.device_capacity_bytes;
    out.stats_.host_object_count     = plan.host_objects.size();
    // A shard's plane alignment gaps are bytes no copy writes; they must read as zero, like the
    // padding of a stored parent. A device holding a shard therefore gets a ZeroFill::Yes arena,
    // zeroed before the constructor returns and so before any shard upload starts. Arenas without
    // shards (every arena at tp 1) are not zeroed, as upstream.
    std::array<bool, kMaximumDevices> holds_shard{};
    for (const auto& placement : plan.device_objects) {
        if (is_sharded(placement.axis) && placement.device >= 0 &&
            placement.device < plan.device_count) {
            holds_shard[static_cast<std::size_t>(placement.device)] = true;
        }
    }
    std::uint64_t capacity = 0;
    for (std::size_t device = 0; device < devices.size(); ++device) {
        const auto bytes = plan.per_device_capacity_bytes[device];
        capacity         = checked_add(capacity, bytes, "device capacity");
        if (bytes > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError("device backing exceeds size_t");
        }
        out.stats_.per_device_capacity_bytes[device] = bytes;
        if (bytes) {
            selection.select(device);
            try {
                out.arenas_[device] = std::make_unique<DeviceArena>(
                    static_cast<std::size_t>(bytes),
                    holds_shard[device] ? ZeroFill::Yes : ZeroFill::No);
            } catch (const std::exception& error) {
                throw ArtifactError("device " + std::to_string(devices[device]->device) +
                                    " weight arena of " + std::to_string(bytes) +
                                    " bytes: " + error.what());
            }
        }
    }
    if (capacity != plan.device_capacity_bytes) {
        throw ArtifactError("device capacity differs from its per-device sum");
    }
    for (auto& placement : plan.host_objects) {
        reader.validate_object(placement.object);
        auto& storage = out.objects_.at(placement.object.index);
        if (!storage.host_data.empty()) { throw ArtifactError("duplicate Host placement"); }
        const auto& object = reader.directory().object(placement.object);
        if (placement.data.empty()) {
            placement.data = reader.read_object(placement.object);
            out.stats_.read_bytes =
                checked_add(out.stats_.read_bytes, placement.data.size(), "Host read bytes");
        }
        if (placement.data.size() != object_bytes(object)) {
            throw ArtifactError("Host placement size differs from object");
        }
        storage.host_data = std::move(placement.data);
        out.stats_.retained_host_bytes =
            checked_add(out.stats_.retained_host_bytes, storage.host_data.size(), "retained bytes");
        if (std::holds_alternative<TensorObject>(object)) {
            const auto& geometry = reader.geometry(placement.object);
            const auto divisor =
                read_divisor(reader, placement.object, geometry, storage.host_data, out.stats_);
            storage.host = WeightParent{geometry, storage.host_data.data(), divisor};
        }
    }
    if (!plan.host_mapped_objects.empty()) {
        // One mapped page-locked allocation for every device: the parents' bytes are read into
        // it once and never uploaded, and each device's parent is that device's address of them.
        if (plan.host_mapped_bytes > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError("host-mapped backing exceeds size_t");
        }
        require_host_memory(plan.host_mapped_bytes);
        try {
            out.host_mapped_ = std::make_unique<PinnedHostBuffer>(
                static_cast<std::size_t>(plan.host_mapped_bytes), HostMapping::Yes,
                PinnedHostPages::Huge);
        } catch (const std::exception& error) {
            throw ArtifactError("host-mapped weight buffer of " +
                                std::to_string(plan.host_mapped_bytes) + " bytes: " + error.what());
        }
        auto* host = static_cast<std::byte*>(out.host_mapped_->data());
        reader.open_all(); // parallel_read below must not race a lazy continuation open
        std::array<const std::byte*, kMaximumDevices> mapped{};
        for (std::size_t device = 0; device < devices.size(); ++device) {
            selection.select(device);
            mapped[device] = static_cast<const std::byte*>(out.host_mapped_->device_data());
        }
        std::uint64_t end = 0;
        std::vector<HostRead> reads;
        reads.reserve(plan.host_mapped_objects.size());
        std::vector<bool> placed(out.objects_.size(), false);
        for (const auto& placement : plan.host_mapped_objects) {
            reader.validate_object(placement.object);
            const auto& geometry   = reader.geometry(placement.object);
            const auto& descriptor = reader.directory().tensor(placement.object);
            const auto& object     = out.objects_.at(placement.object.index);
            if (placement.bytes != geometry.bytes || descriptor.bytes > placement.bytes ||
                !placement.alignment || placement.offset < end ||
                placement.offset % placement.alignment ||
                checked_add(placement.offset, placement.bytes, "host-mapped end") >
                    plan.host_mapped_bytes ||
                placed.at(placement.object.index) ||
                std::any_of(object.device.begin(), object.device.end(),
                            [](const auto& backing) { return backing.parent.has_value(); })) {
                throw ArtifactError("invalid or duplicate host-mapped placement");
            }
            placed[placement.object.index] = true;
            end                = placement.offset + placement.bytes;
            const auto bytes   = std::span(host + placement.offset,
                                           static_cast<std::size_t>(placement.bytes));
            const auto payload = static_cast<std::size_t>(descriptor.bytes);
            reads.push_back({descriptor.offset, bytes.first(payload)});
            // Padding after the stored payload reads as zero, as in a device arena.
            std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(payload), bytes.end(),
                      std::byte{0});
            out.stats_.read_bytes =
                checked_add(out.stats_.read_bytes, descriptor.bytes, "host-mapped read bytes");
        }
        parallel_read(reader, reads, [&](std::uint64_t read) { phase.progress(read, all_total); });
        for (const auto& placement : plan.host_mapped_objects) {
            const auto& geometry = reader.geometry(placement.object);
            auto& object         = out.objects_.at(placement.object.index);
            const auto bytes =
                std::span(host + placement.offset, static_cast<std::size_t>(placement.bytes));
            const auto divisor =
                read_divisor(reader, placement.object, geometry, bytes, out.stats_);
            for (std::size_t device = 0; device < devices.size(); ++device) {
                auto& backing  = object.device[device];
                backing.parent = WeightParent{geometry, mapped[device] + placement.offset, divisor};
                backing.shard  = {ShardAxis::HostMapped, {}, geometry.shape};
                out.stats_.per_device_host_mapped_bytes[device] += placement.bytes;
            }
        }
        out.stats_.host_mapped_bytes = plan.host_mapped_bytes;
    }
    for (const auto handle : plan.file_objects) {
        reader.validate_object(handle);
        auto& storage      = out.objects_.at(handle.index);
        const auto& object = reader.directory().object(handle);
        if (!storage.file.empty()) { throw ArtifactError("duplicate file-mapped placement"); }
        for (const auto& segment : reader.segments(object_offset(object), object_bytes(object))) {
            auto mapping = std::make_unique<FileMapping>(reader.file_path(segment.file_index),
                                                         segment.file_offset, segment.bytes);
            storage.file.push_back({segment.destination_offset, segment.bytes, mapping->data(),
                                    reader.file_path(segment.file_index), segment.file_offset});
            out.mappings_.push_back(std::move(mapping));
        }
        if (std::holds_alternative<TensorObject>(object)) {
            storage.file_geometry = reader.geometry(handle);
        }
        out.stats_.mapped_file_bytes =
            checked_add(out.stats_.mapped_file_bytes, object_bytes(object), "mapped bytes");
    }
    std::vector<CopyRange> ranges;
    const auto add_ranges = [&](const TensorObject& descriptor, std::uint64_t source,
                                std::uint64_t bytes, std::byte* destination, std::size_t device) {
        for (const auto& segment :
             reader.segments(checked_add(descriptor.offset, source, "copy source"), bytes)) {
            ranges.push_back({segment.file_index, segment.file_offset,
                              checked_add(segment.file_offset, segment.bytes, "copy range"),
                              destination + segment.destination_offset, device});
        }
    };
    for (const auto& placement : plan.device_objects) {
        if (placement.device < 0 || placement.device >= plan.device_count) {
            throw ArtifactError("device placement names a device outside the plan");
        }
        const auto device    = static_cast<std::size_t>(placement.device);
        const auto& geometry = reader.geometry(placement.object);
        auto& object         = out.objects_.at(placement.object.index);
        auto& backing        = object.device[device];
        auto& arena          = out.arenas_[device];
        const bool sharded   = is_sharded(placement.axis);
        std::optional<TensorSlice> slice;
        if (sharded) { slice = tensor_slice(geometry, placement.axis, placement.ranges); }
        const auto& resident = slice ? slice->geometry : geometry;
        if (backing.parent || !arena || resident.bytes != placement.bytes ||
            (slice ? slice->copies != placement.copies
                   : !placement.ranges.empty() || !placement.copies.empty())) {
            throw ArtifactError("invalid or duplicate device placement");
        }
        auto storage      = arena->alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                               static_cast<std::size_t>(placement.alignment));
        const auto offset = static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                                       static_cast<std::byte*>(arena->base()));
        if (offset != placement.offset) {
            throw ArtifactError("device offset differs from materialization plan");
        }
        // One divisor read per parent, shared by every device that holds it.
        std::optional<float> divisor;
        if (object.host) { divisor = object.host->weight_scale_divisor; }
        for (const auto& other : object.device) {
            if (!divisor && other.parent) { divisor = other.parent->weight_scale_divisor; }
        }
        if (!divisor) {
            divisor = read_divisor(reader, placement.object, geometry, {}, out.stats_);
        }
        backing.parent =
            WeightParent{resident, static_cast<const std::byte*>(storage.data), *divisor};
        backing.shard          = {placement.axis, placement.ranges, geometry.shape};
        auto* destination      = static_cast<std::byte*>(storage.data);
        const auto& descriptor = reader.directory().tensor(placement.object);
        if (sharded) {
            for (const auto& copy : placement.copies) {
                add_ranges(descriptor, copy.source_offset, copy.bytes,
                           destination + copy.dest_offset, device);
            }
            out.stats_.sharded_bytes[device] += placement.bytes;
        } else {
            add_ranges(descriptor, 0, descriptor.bytes, destination, device);
            auto& kind = placement.axis == ShardAxis::Replicated ? out.stats_.replicated_bytes
                                                                 : out.stats_.local_bytes;
            kind[device] += placement.bytes;
        }
    }
    for (const auto& object : out.objects_) {
        if (std::any_of(object.device.begin(), object.device.end(),
                        [](const auto& backing) { return backing.parent.has_value(); })) {
            ++out.stats_.device_object_count;
        }
    }
    if (ranges.empty()) {
        phase.complete(host_total, all_total);
        return out;
    }
    // Every device byte is written once. Sources may repeat: each device reads its own copy of
    // a replicated parent, and all copies are cut from the staged chunk independently below.
    {
        std::vector<const CopyRange*> written;
        written.reserve(ranges.size());
        for (const auto& range : ranges) { written.push_back(&range); }
        std::sort(written.begin(), written.end(), [](const auto* a, const auto* b) {
            return std::tie(a->device, a->destination) < std::tie(b->device, b->destination);
        });
        for (std::size_t i = 1; i < written.size(); ++i) {
            const auto& previous = *written[i - 1];
            if (previous.device == written[i]->device &&
                previous.destination + (previous.end - previous.begin) > written[i]->destination) {
                throw ArtifactError("device destination ranges overlap");
            }
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        return std::tie(a.file, a.begin) < std::tie(b.file, b.begin);
    });
    std::vector<ReadSpan> spans;
    std::uint64_t aligned_bytes = 0;
    for (const auto& range : ranges) {
        const auto begin = range.begin / kPayloadAlignment * kPayloadAlignment;
        if (spans.empty() || spans.back().file != range.file ||
            begin > align_up(spans.back().end, kPayloadAlignment, "direct range")) {
            spans.push_back({range.file, begin, range.end});
        } else {
            spans.back().end = std::max(spans.back().end, range.end);
        }
    }
    for (const auto& span : spans) {
        aligned_bytes = checked_add(
            aligned_bytes, align_up(span.end - span.begin, kPayloadAlignment, "direct range bytes"),
            "direct bytes");
    }
    const auto slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_bytes));
    const auto slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_bytes - 1) / slot_bytes));
    std::vector<std::unique_ptr<Slot>> slots;
    out.stats_.peak_staging_bytes = slot_bytes * slot_count;
    StartupPhaseScope pin_phase(observer, StartupPhase::WeightsStagingPin,
                                StartupProgressUnit::Bytes, out.stats_.peak_staging_bytes);
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes, selection));
    }
    pin_phase.complete();
    TransferCompletion completion{devices};
    std::size_t next_slot = 0;
    // Ranges before `live` ended before the current chunk. Later ranges may start inside a
    // range that is still being copied, so each chunk scans from `live` to its end.
    std::size_t live     = 0;
    std::uint64_t copied = 0;
    const auto start     = std::chrono::steady_clock::now();
    for (const auto& span : spans) {
        for (auto source = span.begin; source < span.end; source += slot_bytes) {
            auto& slot = *slots[next_slot++ % slot_count];
            slot.wait();
            const auto remaining = span.end - source;
            const auto request   = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes, align_up(remaining, kPayloadAlignment, "direct block bytes")));
            const auto received  = reader.read_direct(
                span.file, source, {slot.data, request});
            if (received < std::min<std::uint64_t>(request, remaining)) {
                throw ArtifactError("direct read ended before the required payload");
            }
            out.stats_.read_bytes = checked_add(out.stats_.read_bytes, received, "read bytes");
            const auto chunk_end  = checked_add(source, received, "read block end");
            while (live < ranges.size() &&
                   (ranges[live].file < span.file ||
                    (ranges[live].file == span.file && ranges[live].end <= source))) {
                ++live;
            }
            for (std::size_t device = 0; device < devices.size(); ++device) {
                bool fed = false;
                for (auto i = live; i < ranges.size() && ranges[i].file == span.file &&
                                    ranges[i].begin < chunk_end;
                     ++i) {
                    const auto& range = ranges[i];
                    const auto begin  = std::max(source, range.begin);
                    const auto end    = std::min(chunk_end, range.end);
                    if (range.device != device || begin >= end) { continue; }
                    if (!fed) { selection.select(device); }
                    check_cuda(cudaMemcpyAsync(range.destination + (begin - range.begin),
                                               static_cast<const std::byte*>(slot.data) +
                                                   (begin - source),
                                               static_cast<std::size_t>(end - begin),
                                               cudaMemcpyHostToDevice,
                                               devices[device]->transfer_stream),
                               "upload weight bytes");
                    copied = checked_add(copied, end - begin, "copied bytes");
                    out.stats_.per_device_h2d_bytes[device] += end - begin;
                    fed = true;
                }
                if (fed) {
                    check_cuda(
                        cudaEventRecord(slot.events[device], devices[device]->transfer_stream),
                        "record weight staging completion");
                    slot.pending[device] = true;
                }
            }
            phase.progress(host_total + copied, all_total);
        }
    }
    for (const auto& slot : slots) { slot->wait(); }
    completion.finish();
    if (copied != total) { throw ArtifactError("incomplete device upload"); }
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    slots.clear();
    phase.complete(host_total + copied, all_total);
    return out;
}

MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                 DeviceContext& device, const StartupObserver* startup_observer) {
    if (plan.device_count != 1) {
        throw ArtifactError("a multi-device materialization plan requires an ExecutionContext");
    }
    DeviceContext* const devices[] = {&device};
    return materialize(reader, std::move(plan), devices, startup_observer);
}

MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                 ExecutionContext& execution,
                                 const StartupObserver* startup_observer) {
    if (plan.device_count != execution.tp) {
        throw ArtifactError("materialization plan device count differs from the tensor-parallel "
                            "degree");
    }
    std::array<DeviceContext*, kMaximumDevices> devices{};
    for (int i = 0; i < execution.tp; ++i) {
        devices[static_cast<std::size_t>(i)] = &*execution.dev[static_cast<std::size_t>(i)];
    }
    return materialize(
        reader, std::move(plan),
        std::span<DeviceContext* const>(devices.data(), static_cast<std::size_t>(execution.tp)),
        startup_observer);
}

} // namespace ninfer::artifact
