#pragma once

// Router-ahead expert prefetch of the Qwen4Exp offload backend (T25; predictor from
// T19, both in docs/capy-flash/development-log.md; staging protocol in core/expert_staging.h).
//
// Per decode window, for every Text layer l whose successor j = l+1 is a prefetch target:
//   * device, right after layer l's attention output is committed (the input of l's MLP
//     hyper-connection, before l's router): ops::moe_predict_ahead with layer j's MLP
//     hyper-connection norm and router publishes j's top non-resident candidates and a flag to
//     mapped memory (T19 `post_uni`);
//   * host (the execution thread, which already follows the window layer by layer): on the flag it
//     takes the first candidates the link can move before layer j's experts run (budget from the
//     measured layer period and the copies still in flight) and copies each into one of layer j's
//     staging slots on a dedicated stream, the slot's tag written after the copy;
//   * device, layer j right after its router: core staging_overlay builds j's effective slot row
//     (the ExpertCache table row + every staged expert whose copy has landed by then); the CPU
//     doorbell and the expert kernel both use it, so a landed expert runs on the GPU (W4A4) and a
//     late one on the CPU as before.
// The staging slots are 2 * slots_per_layer device slots after the ExpertCache pool: target
// layers of one parity share one set (a set is rewritten for layer j+2 only after the device has
// left layer j). They never enter the ExpertCache table, so its per-window consistency is
// unchanged; a staged expert serves only the window and layer it was copied for.
// Counters: issued (copies), landed (tag present when the layer's overlay ran), used (distinct
// routed experts served from staging) and used_routes.

#include "core/expert_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

struct Qwen4PrefetchOptions {
    std::int32_t slots_per_layer = 0;    // copies per target layer and window (0 = off)
    std::int32_t candidates      = 16;   // candidates the predictor publishes per layer (<= 32)
    float min_score              = 0.0F; // window-score gate (sum over the window's tokens of p)
    std::int32_t first_layer     = 2;    // first target layer (layer 1 follows the PLE injection)
    double link_gbps             = 24.0; // copy rate the budget assumes (GB/s)
    double lead                  = 1.0;  // share of one layer period the copies may take
};

struct Qwen4PrefetchCounters {
    std::int64_t windows     = 0;
    std::int64_t predicted   = 0; // layers whose candidates were read
    std::int64_t candidates  = 0; // valid candidates read
    std::int64_t issued      = 0; // experts copied into staging
    std::int64_t landed      = 0; // copies whose tag was there when their layer looked
    std::int64_t used        = 0; // distinct routed experts served from staging
    std::int64_t used_routes = 0; // (token, top-k) routes served from staging
    std::int64_t budget_cut  = 0; // candidates left out by the time budget
};

class Qwen4ExpertPrefetch {
public:
    struct DeviceMemory {
        std::uint32_t* tags        = nullptr; // [2 * slots_per_layer]
        std::uint32_t* epoch       = nullptr; // one word
        std::int32_t* rows         = nullptr; // [text_layers * 512] effective slot rows
        void* predict_workspace    = nullptr; // ops::moe_predict_workspace_bytes()
    };

    // `layout` / `pool` = the ExpertCache's (bank bases of the Text layers first); the staging
    // slots are pool slots [first_slot, first_slot + 2 * slots_per_layer).
    Qwen4ExpertPrefetch(const Qwen4PrefetchOptions& options, std::int32_t text_layers,
                        const expert_cache::ExpertLayout& layout,
                        const std::array<std::byte*, 4>& pool, std::int32_t first_slot,
                        DeviceMemory memory, int device, cudaStream_t stream);
    ~Qwen4ExpertPrefetch();
    Qwen4ExpertPrefetch(const Qwen4ExpertPrefetch&)            = delete;
    Qwen4ExpertPrefetch& operator=(const Qwen4ExpertPrefetch&) = delete;

    [[nodiscard]] static std::int32_t staging_slots(const Qwen4PrefetchOptions& options) noexcept {
        return 2 * options.slots_per_layer;
    }
    // Layer `layer` is a target (its experts may be staged).
    [[nodiscard]] bool stages(std::int32_t layer) const noexcept {
        return layer >= options_.first_layer && layer < text_layers_;
    }
    [[nodiscard]] bool predicts(std::int32_t layer) const noexcept { return stages(layer + 1); }

    // ---- Enqueue (compute stream; CUDA Graph capturable) ----
    // After layer l's attention: candidates of layer l+1. wide BF16 [4*2560, T <= 8]; norm = layer
    // l+1's MLP hyper-connection norm; router = layer l+1's router; resident = l+1's table row.
    void predict(std::int32_t layer, const Tensor& wide, const Tensor& norm, const Tensor& router,
                 float eps, const std::int32_t* resident, cudaStream_t stream);
    // Right after `layer`'s router: its effective slot row (returned, device I32 [512]).
    [[nodiscard]] const std::int32_t* overlay(std::int32_t layer, const std::int32_t* table,
                                              const std::int32_t* ids, std::int32_t routes,
                                              cudaStream_t stream);

    // ---- Window (execution thread) ----
    void begin_window(cudaStream_t stream);
    // True once layer l's candidates are published in this window.
    [[nodiscard]] bool ready(std::int32_t layer) const noexcept;
    // Copies layer l+1's candidates (call once per window after ready(l)); `now_ns` = when the
    // flag was seen (layer period estimate).
    void issue(std::int32_t layer, std::int64_t now_ns);
    // After the device completed the window: reads the per-layer staging statistics.
    void end_window();

    [[nodiscard]] const Qwen4PrefetchCounters& counters() const noexcept { return counters_; }
    [[nodiscard]] double layer_period_us() const noexcept { return period_us_; }
    [[nodiscard]] double rate_scale() const noexcept { return rate_scale_; }

private:
    struct Mapped {
        std::uint32_t* flags     = nullptr; // [text_layers]
        std::int32_t* candidates = nullptr; // [text_layers * 32]
        float* scores            = nullptr; // [text_layers * 32]
        std::uint32_t* stats     = nullptr; // [text_layers * 4]
    };
    struct InFlight {
        cudaEvent_t done   = nullptr;
        std::int32_t count = 0;
    };
    [[nodiscard]] std::int32_t pending_copies();

    Qwen4PrefetchOptions options_;
    std::int32_t text_layers_ = 0;
    expert_cache::ExpertLayout layout_;
    std::array<std::byte*, 4> pool_{};
    std::int32_t first_slot_ = 0;
    DeviceMemory memory_;
    int device_ = 0;
    std::int64_t expert_bytes_ = 0;
    double copy_us_            = 0;

    std::byte* host_ = nullptr; // mapped allocation
    Mapped mapped_;
    cudaStream_t copy_stream_ = nullptr;
    std::deque<InFlight> in_flight_;
    std::vector<cudaEvent_t> spare_events_;

    std::uint32_t epoch_ = 0;
    std::vector<std::uint8_t> issued_layer_; // [text_layers] this window
    std::int64_t first_seen_ns_ = 0, last_seen_ns_ = 0;
    std::int32_t first_seen_layer_ = -1, last_seen_layer_ = -1;
    double period_us_ = 400.0;
    double rate_scale_ = 1.0;          // landing feedback on link_gbps
    std::int64_t window_issued_at_ = 0; // counters_.issued at the previous end_window
    Qwen4PrefetchCounters counters_;

    // Scratch of issue().
    std::vector<void*> dst_;
    std::vector<const void*> src_;
    std::vector<std::size_t> bytes_;
};

} // namespace ninfer::models::qwen3_5::execution
