#pragma once

// ninfer::ops::detail - private launch interface of the fused stream-split hyper-connection mix
// (split_fused.cu; public contract in include/ninfer/ops/tp2/hyper_connection.h).

#include "ops/hyper_connection/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// One rank's raw view of one mailbox slot (ninfer::ops::PeerExchangeSlot without the hang word).
struct HcSplitExchange {
    void* mine_payload        = nullptr;
    const void* peer_payload  = nullptr;
    std::uint32_t* mine_flags = nullptr;
    std::uint32_t* peer_flags = nullptr;
    std::uint32_t* epochs     = nullptr;
};

// One rank's fused split mix over its two streams (residual and weights are the rank's shards).
struct HcSplitFused {
    __nv_bfloat16* residual         = nullptr; // [2H, T], written only with a pending combine
    const __nv_bfloat16* previous_y = nullptr; // [H, T] or nullptr
    const float* previous_inject    = nullptr; // [2, T], the rank's streams
    const __nv_bfloat16* norm       = nullptr; // [2H]
    const std::int8_t* down_codes   = nullptr; // [rows, 2H]
    const __half* down_scales       = nullptr; // [rows, 2H/32]
    const std::int8_t* up_codes     = nullptr; // [2H, 384]
    const __half* up_scales         = nullptr; // [2H, 12]
    __nv_bfloat16* mixed            = nullptr; // [H, T]
    float* inject                   = nullptr; // [2, T] or nullptr
    float* partials                 = nullptr; // workspace [T, kHcRowsCap, 2]
    float* square_sums              = nullptr; // workspace [T, 2]
    float* projection               = nullptr; // workspace [T, rows] (rank partial, then the sum)
    float* collapse                 = nullptr; // workspace [T, H] (rank partial of mixed * S)
    HcSplitExchange exchange1;                 // T * rows * 4 bytes
    HcSplitExchange exchange2;                 // T * H * 4 bytes
    std::uint32_t* hang       = nullptr;
    std::int32_t rows         = 0; // kHcRank or kHcRowsCap
    std::int32_t tokens       = 0;
    std::int32_t first_stream = 0; // global index of the rank's first stream
    float eps                 = 0.0F;
};

// CTAs of the fused split route on the current device (even, >= 56), 0 when it cannot host it.
[[nodiscard]] int hyper_connection_split_fused_grid();
[[nodiscard]] std::size_t hyper_connection_split_fused_workspace_bytes(std::int32_t tokens);
void hyper_connection_split_fused_launch(const HcSplitFused& params, cudaStream_t stream);

} // namespace ninfer::ops::detail
