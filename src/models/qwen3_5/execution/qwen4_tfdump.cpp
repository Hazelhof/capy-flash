#include <algorithm>
#include "models/qwen3_5/execution/qwen4_tfdump.h"

#include "core/device.h"
#include "core/device_scope.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

// .npy v1.0, little-endian, C order, one dimension.
void write_npy(const std::filesystem::path& path, std::string_view descr, std::size_t count,
               const void* data, std::size_t element_bytes) {
    std::ostringstream header;
    header << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (" << count << ",), }";
    std::string text           = header.str();
    const std::size_t unpadded = 10 + text.size() + 1;
    text.append((64 - unpadded % 64) % 64, ' ');
    text.push_back('\n');
    std::ofstream out(path, std::ios::binary);
    if (!out) { throw std::runtime_error("tfdump: cannot write " + path.string()); }
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    out.write(magic, sizeof(magic));
    const auto length          = static_cast<std::uint16_t>(text.size());
    const char length_bytes[2] = {static_cast<char>(length & 0xFF), static_cast<char>(length >> 8)};
    out.write(length_bytes, 2);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(count * element_bytes));
    if (!out) { throw std::runtime_error("tfdump: short write " + path.string()); }
}

std::string position_name(std::int32_t position) {
    char name[32];
    std::snprintf(name, sizeof(name), "p%06d.npy", position);
    return name;
}

} // namespace

void require_qwen4_validation_opt_in() {
    const char* opt_in = std::getenv("NINFER_QWEN4_VALIDATION");
    if (opt_in != nullptr && std::string_view(opt_in) == "1") { return; }
    for (const char* knob : {"NINFER_QWEN4_FORCE_TOKENS", "NINFER_QWEN4_TFDUMP_FORCE",
                             "NINFER_QWEN4_EXPERT_FREEZE", "NINFER_QWEN4_REPLAY"}) {
        const char* value = std::getenv(knob);
        if (value != nullptr && *value != '\0') {
            throw std::invalid_argument(std::string(knob) +
                                        " changes the generated tokens (validation only); set "
                                        "NINFER_QWEN4_VALIDATION=1 to use it");
        }
    }
}

std::unique_ptr<Qwen4TfDump> Qwen4TfDump::from_environment(std::int32_t layers) {
    const char* directory = std::getenv("NINFER_QWEN4_TFDUMP");
    if (directory == nullptr || *directory == '\0') { return nullptr; }
    std::vector<bool> selected(static_cast<std::size_t>(layers), true);
    if (const char* list = std::getenv("NINFER_QWEN4_TFDUMP_LAYERS")) {
        const std::string_view text(list);
        if (text != "all") {
            selected.assign(selected.size(), false);
            if (text != "none") {
                std::stringstream stream{std::string(text)};
                std::string item;
                while (std::getline(stream, item, ',')) {
                    const int layer = std::stoi(item);
                    if (layer < 0 || layer >= layers) {
                        throw std::invalid_argument("NINFER_QWEN4_TFDUMP_LAYERS out of range");
                    }
                    selected[static_cast<std::size_t>(layer)] = true;
                }
            }
        }
    }
    const char* logits = std::getenv("NINFER_QWEN4_TFDUMP_LOGITS");
    auto dump          = std::make_unique<Qwen4TfDump>(directory, std::move(selected),
                                                       logits == nullptr || std::string_view(logits) != "0");
    if (const char* force = std::getenv("NINFER_QWEN4_TFDUMP_FORCE")) {
        std::ifstream in(force);
        if (!in) {
            throw std::invalid_argument("NINFER_QWEN4_TFDUMP_FORCE: cannot read the ids file");
        }
        std::vector<std::int32_t> ids;
        long long value = 0;
        while (in >> value) { ids.push_back(static_cast<std::int32_t>(value)); }
        dump->set_forced(std::move(ids));
    }
    const char* mtp = std::getenv("NINFER_QWEN4_TFDUMP_MTP");
    dump->set_mtp_logits(mtp != nullptr && std::string_view(mtp) == "1");
    if (const char* stride = std::getenv("NINFER_QWEN4_TFDUMP_STRIDE")) {
        dump->stride_ = std::max(1, std::atoi(stride));
    }
    if (const char* x = std::getenv("NINFER_QWEN4_XDUMP"); x != nullptr && *x != '\0') {
        dump->xdir_ = x;
        std::filesystem::create_directories(dump->xdir_);
        if (const char* xs = std::getenv("NINFER_QWEN4_XDUMP_STRIDE")) {
            dump->xstride_ = std::max(1, std::atoi(xs));
        }
    }
    if (const char* sequences = std::getenv("NINFER_QWEN4_TFDUMP_SEQUENCES")) {
        dump->sequences_ = std::string_view(sequences) == "1";
    }
    if (const char* compact = std::getenv("NINFER_QWEN4_TFDUMP_COMPACT")) {
        dump->compact_ = std::string_view(compact) == "1";
    }
    if (const char* topk = std::getenv("NINFER_QWEN4_TFDUMP_TOPK")) {
        dump->topk_ = std::max(0, std::atoi(topk));
    }
    return dump;
}

Qwen4TfDump::Qwen4TfDump(std::filesystem::path directory, std::vector<bool> layers, bool logits)
    : directory_(std::move(directory)), layers_(std::move(layers)), logits_(logits) {
    std::filesystem::create_directories(directory_);
}

bool Qwen4TfDump::wants(std::int32_t layer) const {
    if (layer < 0) { return true; }
    return static_cast<std::size_t>(layer) < layers_.size() &&
           layers_[static_cast<std::size_t>(layer)];
}

void Qwen4TfDump::capture(std::int32_t layer, const Tensor& values, std::int32_t first_position,
                          cudaStream_t stream) {
    if (values.dtype != DType::BF16 || !values.is_contiguous()) {
        throw std::invalid_argument("tfdump: taps take contiguous BF16 [rows, T]");
    }
    const auto rows    = static_cast<std::size_t>(values.ne[0]);
    const auto columns = static_cast<std::size_t>(values.numel()) / rows;
    scratch_.resize(rows * columns);
    CUDA_CHECK(cudaMemcpyAsync(scratch_.data(), values.data, values.bytes(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    char name[16];
    if (layer < 0) {
        std::snprintf(name, sizeof(name), "final");
    } else {
        std::snprintf(name, sizeof(name), "L%02d", layer);
    }
    if (compact_) {
        std::filesystem::create_directories(out_dir());
        std::ofstream out(out_dir() / ("hidden_" + std::string(name) + ".bin"),
                          std::ios::binary | std::ios::app);
        const std::int32_t head[3] = {first_position, static_cast<std::int32_t>(columns),
                                      static_cast<std::int32_t>(rows)};
        out.write(reinterpret_cast<const char*>(head), sizeof(head));
        out.write(reinterpret_cast<const char*>(scratch_.data()),
                  static_cast<std::streamsize>(scratch_.size() * sizeof(std::uint16_t)));
        if (!out) { throw std::runtime_error("tfdump: short write of a compact hidden tap"); }
        return;
    }
    const auto directory = out_dir() / "hidden" / name;
    std::filesystem::create_directories(directory);
    std::vector<float> column(rows);
    for (std::size_t c = 0; c < columns; ++c) {
        for (std::size_t r = 0; r < rows; ++r) {
            column[r] = bf16_to_float(scratch_[c * rows + r]);
        }
        write_npy(directory / position_name(first_position + static_cast<std::int32_t>(c)), "<f4",
                  rows, column.data(), sizeof(float));
    }
}

void Qwen4TfDump::compare_ranks(std::int32_t layer, const Tensor& rank0, const Tensor& rank1,
                                cudaStream_t stream0, cudaStream_t stream1) {
    if (rank0.bytes() != rank1.bytes() || rank0.dtype != DType::BF16) {
        throw std::invalid_argument("tfdump: rank taps differ in shape");
    }
    const auto count = static_cast<std::size_t>(rank0.numel());
    std::vector<std::uint16_t> first(count), second(count);
    // Each copy on its rank's stream, with that stream's device current.
    const ScopedCurrentDevice restore;
    const std::array<std::pair<const Tensor*, cudaStream_t>, 2> ranks{std::pair{&rank0, stream0},
                                                                      std::pair{&rank1, stream1}};
    std::array<std::uint16_t*, 2> host{first.data(), second.data()};
    for (std::size_t r = 0; r < 2; ++r) {
        int device = 0;
        CUDA_CHECK(cudaStreamGetDevice(ranks[r].second, &device));
        ScopedCurrentDevice::select(device);
        CUDA_CHECK(cudaMemcpyAsync(host[r], ranks[r].first->data, ranks[r].first->bytes(),
                                   cudaMemcpyDeviceToHost, ranks[r].second));
        CUDA_CHECK(cudaStreamSynchronize(ranks[r].second));
    }
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) { mismatches += first[i] != second[i] ? 1U : 0U; }
    std::ofstream out(directory_ / "rank_check.txt", std::ios::app);
    out << "layer " << layer << " columns " << count / static_cast<std::size_t>(rank0.ne[0])
        << " bf16_mismatches " << mismatches << " of " << count << '\n';
}

void Qwen4TfDump::capture_exchange(std::int32_t layer, int site, const Tensor& rank0,
                                   const Tensor& rank1, std::int32_t first_position,
                                   cudaStream_t stream0, cudaStream_t stream1) {
    const bool fp32 = rank0.dtype == DType::FP32;
    if (rank0.bytes() != rank1.bytes() || rank0.dtype != rank1.dtype || !rank0.is_contiguous() ||
        !rank1.is_contiguous() || (!fp32 && rank0.dtype != DType::BF16)) {
        throw std::invalid_argument("xdump: rank partials must be matching contiguous BF16/FP32");
    }
    const auto rows           = static_cast<std::size_t>(rank0.ne[0]);
    const auto columns        = static_cast<std::size_t>(rank0.numel()) / rows;
    const std::size_t element = fp32 ? 4U : 2U;
    std::vector<std::int32_t> keep;
    for (std::size_t c = 0; c < columns; ++c) {
        const auto position = first_position + static_cast<std::int32_t>(c);
        if (position % xstride_ == 0) { keep.push_back(static_cast<std::int32_t>(c)); }
    }
    if (keep.empty()) { return; }
    std::vector<std::byte> host(rank0.bytes()), out(keep.size() * rows * element);
    const ScopedCurrentDevice restore;
    const std::array<std::pair<const Tensor*, cudaStream_t>, 2> ranks{std::pair{&rank0, stream0},
                                                                      std::pair{&rank1, stream1}};
    char name[64];
    for (std::size_t r = 0; r < 2; ++r) {
        int device = 0;
        CUDA_CHECK(cudaStreamGetDevice(ranks[r].second, &device));
        ScopedCurrentDevice::select(device);
        CUDA_CHECK(cudaMemcpyAsync(host.data(), ranks[r].first->data, ranks[r].first->bytes(),
                                   cudaMemcpyDeviceToHost, ranks[r].second));
        CUDA_CHECK(cudaStreamSynchronize(ranks[r].second));
        for (std::size_t k = 0; k < keep.size(); ++k) {
            std::memcpy(out.data() + k * rows * element,
                        host.data() + static_cast<std::size_t>(keep[k]) * rows * element,
                        rows * element);
        }
        std::snprintf(name, sizeof(name), "L%02d_%s_p%06d_r%zu.npy", layer,
                      site == 0 ? "mix" : "moe", first_position, r);
        write_npy(xdir_ / name, fp32 ? "<f4" : "<u2", keep.size() * rows, out.data(), element);
    }
    std::ofstream index(xdir_ / "index.tsv", std::ios::app);
    index << layer << '\t' << (site == 0 ? "mix" : "moe") << '\t' << first_position << '\t' << rows
          << '\t' << keep.size() << '\t' << (fp32 ? "f32" : "bf16") << '\t' << keep.front() << '\n';
}

void Qwen4TfDump::capture_mtp_logits(const Tensor& logits, std::int32_t position,
                                     cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || logits.ne[1] != 1 || !logits.is_contiguous()) {
        throw std::invalid_argument("tfdump: MTP logits are one contiguous BF16 column");
    }
    const auto rows = static_cast<std::size_t>(logits.ne[0]);
    scratch_.resize(rows);
    CUDA_CHECK(cudaMemcpyAsync(scratch_.data(), logits.data, logits.bytes(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto directory = directory_ / "mtp_logits";
    std::filesystem::create_directories(directory);
    std::vector<float> column(rows);
    for (std::size_t r = 0; r < rows; ++r) { column[r] = bf16_to_float(scratch_[r]); }
    write_npy(directory / position_name(position), "<f4", rows, column.data(), sizeof(float));
}

void Qwen4TfDump::write_tokens(std::span<const std::int32_t> tokens) const {
    if (sequences_) {
        char name[16];
        std::snprintf(name, sizeof(name), "seq%03d", ++sequence_);
        current_ = directory_ / name;
        std::filesystem::create_directories(current_);
    }
    std::vector<std::int64_t> wide(tokens.begin(), tokens.end());
    write_npy(out_dir() / "tokens.npy", "<i8", wide.size(), wide.data(), sizeof(std::int64_t));
}

void Qwen4TfDump::write_logits(std::span<const std::uint16_t> logits, std::int32_t rows,
                               std::span<const std::int32_t> positions) const {
    if (!logits_) { return; }
    const auto r = static_cast<std::size_t>(rows);
    if (logits.size() < r * positions.size()) {
        throw std::invalid_argument("tfdump: logits do not cover their positions");
    }
    if (topk_ > 0) {
        std::filesystem::create_directories(out_dir());
        std::ofstream out(out_dir() / "topk.bin", std::ios::binary | std::ios::app);
        const auto k = static_cast<std::size_t>(std::min<std::int32_t>(topk_, rows));
        std::vector<float> column(r);
        std::vector<std::int32_t> order(r);
        std::vector<float> lp(k);
        for (std::size_t c = 0; c < positions.size(); ++c) {
            float top = -INFINITY;
            for (std::size_t i = 0; i < r; ++i) {
                column[i] = bf16_to_float(logits[c * r + i]);
                top       = std::max(top, column[i]);
            }
            double sum = 0;
            for (std::size_t i = 0; i < r; ++i) {
                sum += std::exp(static_cast<double>(column[i] - top));
            }
            const float lse = top + static_cast<float>(std::log(sum));
            for (std::size_t i = 0; i < r; ++i) { order[i] = static_cast<std::int32_t>(i); }
            std::partial_sort(
                order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                [&](std::int32_t a, std::int32_t b) { return column[a] > column[b]; });
            for (std::size_t j = 0; j < k; ++j) { lp[j] = column[order[j]] - lse; }
            out.write(reinterpret_cast<const char*>(&positions[c]), sizeof(std::int32_t));
            out.write(reinterpret_cast<const char*>(order.data()),
                      static_cast<std::streamsize>(k * sizeof(std::int32_t)));
            out.write(reinterpret_cast<const char*>(lp.data()),
                      static_cast<std::streamsize>(k * sizeof(float)));
        }
        if (!out) { throw std::runtime_error("tfdump: short write of the top-k logits"); }
        return;
    }
    const auto directory = out_dir() / "logits";
    std::filesystem::create_directories(directory);
    std::vector<float> column(r);
    for (std::size_t c = 0; c < positions.size(); ++c) {
        if (positions[c] % stride_ != 0) { continue; }
        for (std::size_t i = 0; i < r; ++i) { column[i] = bf16_to_float(logits[c * r + i]); }
        write_npy(directory / position_name(positions[c]), "<f4", r, column.data(), sizeof(float));
    }
}

void Qwen4TfDump::write_meta(const std::string& producer, const std::string& artifact) const {
    std::ofstream out(directory_ / "meta.json");
    out << "{\"format\":\"tfdump/1\",\"producer\":\"" << producer << "\",\"artifact\":\""
        << artifact << "\",\"logits\":\"bf16 output head widened to f32\"}\n";
}

bool qwen4_replay_enabled() {
    static const bool enabled = [] {
        const char* path = std::getenv("NINFER_QWEN4_REPLAY");
        return path != nullptr && *path != '\0';
    }();
    return enabled;
}

std::unique_ptr<Qwen4Replay> Qwen4Replay::from_environment() {
    if (!qwen4_replay_enabled()) { return nullptr; }
    return std::make_unique<Qwen4Replay>(std::getenv("NINFER_QWEN4_REPLAY"));
}

Qwen4Replay::Qwen4Replay(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::invalid_argument("NINFER_QWEN4_REPLAY: cannot read " + path); }
    std::int64_t windows = 0;
    for (std::string line; std::getline(in, line);) {
        std::istringstream row(line);
        std::string tag;
        if (!(row >> tag)) { continue; }
        if (tag == "seq") {
            Qwen4ReplaySequence seq;
            for (long long v = 0; row >> v;) { seq.ids.push_back(static_cast<std::int32_t>(v)); }
            if (seq.ids.size() < 2) {
                throw std::invalid_argument("NINFER_QWEN4_REPLAY: short seq line");
            }
            sequences_.push_back(std::move(seq));
        } else if (tag == "win") {
            if (sequences_.empty()) {
                throw std::invalid_argument("NINFER_QWEN4_REPLAY: win before seq");
            }
            std::uint32_t frontier = 0;
            Qwen4ReplayWindow w;
            if (!(row >> frontier >> w.extent >> w.accepted) || w.accepted > w.extent) {
                throw std::invalid_argument("NINFER_QWEN4_REPLAY: bad win line: " + line);
            }
            for (long long v = 0; row >> v;) { w.drafts.push_back(static_cast<std::int32_t>(v)); }
            if (w.drafts.size() < w.extent) {
                throw std::invalid_argument(
                    "NINFER_QWEN4_REPLAY: win line with fewer drafts than its extent");
            }
            const auto& ids = sequences_.back().ids;
            // The accepted drafts are the reference tokens after the anchor (recorded so).
            for (std::uint32_t j = 0; j < w.accepted; ++j) {
                if (frontier + 1 + j >= ids.size() || w.drafts[j] != ids[frontier + 1 + j]) {
                    throw std::invalid_argument(
                        "NINFER_QWEN4_REPLAY: accepted draft differs from the reference: " + line);
                }
            }
            sequences_.back().windows[frontier] = std::move(w);
            ++windows;
        }
    }
    if (sequences_.empty()) { throw std::invalid_argument("NINFER_QWEN4_REPLAY: no seq line"); }
    std::fprintf(stderr, "qwen4 replay: %s, %zu sequences, %lld windows\n", path.c_str(),
                 sequences_.size(), static_cast<long long>(windows));
}

Qwen4Replay::~Qwen4Replay() {
    std::fprintf(
        stderr,
        "qwen4 replay: forced tokens %lld, windows replayed %lld, unlisted %lld, rounds of "
        "unmatched requests %lld, commit mismatches %lld, ordinary argmax != reference %lld\n",
        static_cast<long long>(forced_tokens), static_cast<long long>(windows_replayed),
        static_cast<long long>(windows_unlisted), static_cast<long long>(rounds_unmatched),
        static_cast<long long>(commit_mismatches), static_cast<long long>(argmax_disagreements));
}

const Qwen4ReplaySequence* Qwen4Replay::match(std::span<const std::int32_t> ledger) const {
    const auto fits = [&](const Qwen4ReplaySequence* s) {
        return s != nullptr && s->ids.size() > ledger.size() &&
               std::equal(ledger.begin(), ledger.end(), s->ids.begin());
    };
    if (fits(last_)) { return last_; }
    for (const Qwen4ReplaySequence& s : sequences_) {
        if (fits(&s)) {
            last_ = &s;
            return last_;
        }
    }
    return nullptr;
}

} // namespace ninfer::models::qwen3_5::execution
