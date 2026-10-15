#include "models/qwen3_5/execution/qwen4_prefetch.h"

#include "core/device.h"
#include "core/expert_staging.h"
#include "ninfer/ops/moe_predict.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

constexpr std::int32_t kExperts    = expert_cache::kStagingExperts;
constexpr std::int32_t kCandidates = ops::kMoePredictMaxCandidates;
constexpr std::int32_t kStats      = expert_cache::kStagingStatsWords;

std::size_t round_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

} // namespace

Qwen4ExpertPrefetch::Qwen4ExpertPrefetch(const Qwen4PrefetchOptions& options,
                                         std::int32_t text_layers,
                                         const expert_cache::ExpertLayout& layout,
                                         const std::array<std::byte*, 4>& pool,
                                         std::int32_t first_slot, DeviceMemory memory, int device,
                                         cudaStream_t stream)
    : options_(options), text_layers_(text_layers), layout_(layout), pool_(pool),
      first_slot_(first_slot), memory_(memory), device_(device) {
    if (options_.slots_per_layer < 1 || options_.slots_per_layer > 64 ||
        options_.candidates < 1 || options_.candidates > kCandidates || options_.first_layer < 1 ||
        !(options_.min_score >= 0.0F) || !(options_.link_gbps > 0.0) || !(options_.lead > 0.0)) {
        throw std::invalid_argument("Qwen4Exp expert prefetch: invalid options (slots 1..64, "
                                    "candidates 1..32, first layer >= 1, min score >= 0, rate > 0)");
    }
    if (text_layers_ < 2 || text_layers_ > expert_cache::kMaxStagingLayers || first_slot_ < 0 ||
        layout_.planes != 4 ||
        layout_.bank_base.size() < static_cast<std::size_t>(text_layers_) * 4U) {
        throw std::invalid_argument("Qwen4Exp expert prefetch: invalid layout");
    }
    if (memory_.tags == nullptr || memory_.epoch == nullptr || memory_.rows == nullptr ||
        memory_.predict_workspace == nullptr ||
        std::any_of(pool_.begin(), pool_.end(), [](const std::byte* p) { return p == nullptr; })) {
        throw std::invalid_argument("Qwen4Exp expert prefetch: device memory is missing");
    }
    for (std::int32_t q = 0; q < layout_.planes; ++q) { expert_bytes_ += layout_.plane_bytes[q]; }
    copy_us_ = static_cast<double>(expert_bytes_) / (options_.link_gbps * 1e3);

    // Mapped host block: flags, candidates, scores, staging statistics per Text layer.
    const auto L            = static_cast<std::size_t>(text_layers_);
    const std::size_t flags = round_up(L * sizeof(std::uint32_t), 256);
    const std::size_t cands = round_up(L * kCandidates * sizeof(std::int32_t), 256);
    const std::size_t score = round_up(L * kCandidates * sizeof(float), 256);
    const std::size_t stats = round_up(L * kStats * sizeof(std::uint32_t), 256);
    void* host              = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, flags + cands + score + stats,
                             cudaHostAllocMapped | cudaHostAllocPortable));
    host_ = static_cast<std::byte*>(host);
    std::memset(host_, 0, flags + cands + score + stats);
    mapped_.flags      = reinterpret_cast<std::uint32_t*>(host_);
    mapped_.candidates = reinterpret_cast<std::int32_t*>(host_ + flags);
    mapped_.scores     = reinterpret_cast<float*>(host_ + flags + cands);
    mapped_.stats      = reinterpret_cast<std::uint32_t*>(host_ + flags + cands + score);
    // Unified addressing: the host addresses are the device addresses (checked).
    void* device_view = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&device_view, host_, 0));
    if (device_view != host) {
        throw std::runtime_error("Qwen4Exp expert prefetch: mapped memory needs unified addressing");
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemsetAsync(memory_.tags, 0xFF,
                               sizeof(std::uint32_t) * static_cast<std::size_t>(staging_slots(options_)),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(memory_.epoch, 0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(memory_.predict_workspace, 0, ops::moe_predict_workspace_bytes(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    issued_layer_.assign(L, 0);
}

Qwen4ExpertPrefetch::~Qwen4ExpertPrefetch() {
    if (copy_stream_ != nullptr) { (void)cudaStreamSynchronize(copy_stream_); }
    for (InFlight& f : in_flight_) { (void)cudaEventDestroy(f.done); }
    for (cudaEvent_t e : spare_events_) { (void)cudaEventDestroy(e); }
    if (copy_stream_ != nullptr) { (void)cudaStreamDestroy(copy_stream_); }
    if (host_ != nullptr) { (void)cudaFreeHost(host_); }
}

void Qwen4ExpertPrefetch::predict(std::int32_t layer, const Tensor& wide, const Tensor& norm,
                                  const Tensor& router, float eps, const std::int32_t* resident,
                                  cudaStream_t stream) {
    if (!predicts(layer)) { throw std::logic_error("Qwen4Exp expert prefetch: layer predicts nothing"); }
    const auto l = static_cast<std::size_t>(layer);
    ops::moe_predict_ahead(wide, norm, router, eps, resident, options_.min_score,
                           options_.candidates, mapped_.candidates + l * kCandidates,
                           mapped_.scores + l * kCandidates, mapped_.flags + l,
                           memory_.predict_workspace, stream);
}

const std::int32_t* Qwen4ExpertPrefetch::overlay(std::int32_t layer, const std::int32_t* table,
                                                 const std::int32_t* ids, std::int32_t routes,
                                                 cudaStream_t stream) {
    if (!stages(layer)) { throw std::logic_error("Qwen4Exp expert prefetch: layer is not staged"); }
    const std::int32_t B   = options_.slots_per_layer;
    const std::int32_t set = (layer & 1) * B;
    std::int32_t* out      = memory_.rows + static_cast<std::size_t>(layer) * kExperts;
    expert_cache::staging_overlay(table, memory_.tags + set, B, first_slot_ + set, memory_.epoch,
                                  layer, ids, routes, out,
                                  mapped_.stats + static_cast<std::size_t>(layer) * kStats, stream);
    return out;
}

void Qwen4ExpertPrefetch::begin_window(cudaStream_t stream) {
    if (epoch_ >= expert_cache::kMaxStagingEpoch) {
        // Wrap: no tag of an old window may survive into a new window with the same epoch.
        CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
        CUDA_CHECK(cudaMemsetAsync(
            memory_.tags, 0xFF,
            sizeof(std::uint32_t) * static_cast<std::size_t>(staging_slots(options_)), stream));
        epoch_ = 0;
    }
    ++epoch_;
    expert_cache::write_staging_word(stream, memory_.epoch, epoch_);
    for (std::int32_t l = 0; l < text_layers_; ++l) {
        reinterpret_cast<volatile std::uint32_t*>(mapped_.flags)[l] = 0;
    }
    std::fill(issued_layer_.begin(), issued_layer_.end(), 0);
    first_seen_layer_ = last_seen_layer_ = -1;
}

bool Qwen4ExpertPrefetch::ready(std::int32_t layer) const noexcept {
    return reinterpret_cast<const volatile std::uint32_t*>(mapped_.flags)[layer] != 0;
}

std::int32_t Qwen4ExpertPrefetch::pending_copies() {
    while (!in_flight_.empty()) {
        const cudaError_t status = cudaEventQuery(in_flight_.front().done);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        spare_events_.push_back(in_flight_.front().done);
        in_flight_.pop_front();
    }
    std::int32_t n = 0;
    for (const InFlight& f : in_flight_) { n += f.count; }
    return n;
}

void Qwen4ExpertPrefetch::issue(std::int32_t layer, std::int64_t now_ns) {
    if (!predicts(layer) || issued_layer_[static_cast<std::size_t>(layer)] != 0) { return; }
    issued_layer_[static_cast<std::size_t>(layer)] = 1;
    if (first_seen_layer_ < 0) {
        first_seen_layer_ = layer;
        first_seen_ns_    = now_ns;
    }
    last_seen_layer_ = layer;
    last_seen_ns_    = now_ns;

    const std::int32_t target = layer + 1;
    const std::int32_t* cands = mapped_.candidates + static_cast<std::size_t>(layer) * kCandidates;
    std::int32_t valid        = 0;
    while (valid < options_.candidates && cands[valid] >= 0) { ++valid; }
    ++counters_.predicted;
    counters_.candidates += valid;

    const std::int32_t B = options_.slots_per_layer;
    // Copies still in flight go first on the stream; the rest of one layer period is this
    // layer's (the copy time scaled by the landing feedback of the previous windows).
    const double copy_us = copy_us_ / rate_scale_;
    const double avail   = period_us_ * options_.lead -
                         static_cast<double>(pending_copies()) * copy_us;
    const auto by_time   = static_cast<std::int32_t>(
        std::clamp(std::floor(avail / copy_us), 0.0, static_cast<double>(B)));
    const std::int32_t n = std::min(by_time, valid);
    counters_.budget_cut += std::min(valid, B) - n;
    if (n == 0) { return; }

    const std::int32_t set = (target & 1) * B;
    const std::int32_t planes = layout_.planes;
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder    = cudaMemcpySrcAccessOrderStream;
    attr.flags             = cudaMemcpyFlagPreferOverlapWithCompute;
    std::size_t attr_index = 0;
    for (std::int32_t i = 0; i < n; ++i) {
        const std::int32_t e    = cands[i];
        const std::int32_t slot = first_slot_ + set + i;
        dst_.clear();
        src_.clear();
        bytes_.clear();
        for (std::int32_t q = 0; q < planes; ++q) {
            dst_.push_back(pool_[static_cast<std::size_t>(q)] +
                           static_cast<std::int64_t>(slot) * layout_.pool_stride[q]);
            src_.push_back(layout_.bank_base[static_cast<std::size_t>(target * planes + q)] +
                           static_cast<std::int64_t>(e) * layout_.bank_stride[q]);
            bytes_.push_back(static_cast<std::size_t>(layout_.plane_bytes[q]));
        }
        CUDA_CHECK(cudaMemcpyBatchAsync(dst_.data(), src_.data(), bytes_.data(), dst_.size(), &attr,
                                        &attr_index, 1, copy_stream_));
        expert_cache::write_staging_word(copy_stream_, memory_.tags + set + i,
                                         expert_cache::staging_tag(epoch_, target, e));
    }
    InFlight f;
    f.count = n;
    if (!spare_events_.empty()) {
        f.done = spare_events_.back();
        spare_events_.pop_back();
    } else {
        CUDA_CHECK(cudaEventCreateWithFlags(&f.done, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(f.done, copy_stream_));
    in_flight_.push_back(f);
    counters_.issued += n;
}

void Qwen4ExpertPrefetch::end_window() {
    ++counters_.windows;
    const std::int64_t issued_before = window_issued_at_, landed_before = counters_.landed;
    window_issued_at_                = counters_.issued;
    const auto* stats = reinterpret_cast<const volatile std::uint32_t*>(mapped_.stats);
    for (std::int32_t j = options_.first_layer; j < text_layers_; ++j) {
        const volatile std::uint32_t* s = stats + static_cast<std::size_t>(j) * kStats;
        if (s[0] != epoch_) { continue; }
        counters_.landed += s[1];
        counters_.used += s[2];
        counters_.used_routes += s[3];
    }
    // Landing feedback: the link is shared with the cache fills and the CPU's reads, so the
    // assumed rate follows what actually landed in time (most copies late: slower; nearly all
    // in time: faster), within [0.25, 2] of link_gbps.
    const std::int64_t issued = counters_.issued - issued_before;
    if (issued >= 8) {
        const double in_time = static_cast<double>(counters_.landed - landed_before) /
                               static_cast<double>(issued);
        if (in_time < 0.75) { rate_scale_ = std::max(0.25, rate_scale_ * 0.9); }
        if (in_time > 0.92) { rate_scale_ = std::min(2.0, rate_scale_ * 1.05); }
    }
    if (first_seen_layer_ >= 0 && last_seen_layer_ > first_seen_layer_) {
        const double period = 1e-3 * static_cast<double>(last_seen_ns_ - first_seen_ns_) /
                              static_cast<double>(last_seen_layer_ - first_seen_layer_);
        period_us_ = 0.5 * period_us_ + 0.5 * period;
    }
}

} // namespace ninfer::models::qwen3_5::execution
