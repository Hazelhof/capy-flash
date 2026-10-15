#pragma once

// ninfer::ops::detail - private launch of the slot-pool NVFP4 MoE (moe_slots_nvfp4).

#include "core/tensor.h"
#include "ninfer/ops/moe_slots.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

struct MoeSlotsWorkspace {
    std::uint32_t* h_codes   = nullptr; // [10T * T][I/8] E2M1 words of h, (first position, token)
    float* h_scales          = nullptr; // [10T * T][I/16]
    float* partial           = nullptr; // [T * 10][H] weighted expert outputs per (token, position)
    std::uint32_t* tile_done = nullptr; // [H/128] arrival counters, cleared by the first kernel
};

std::size_t moe_slots_workspace_layout(std::int32_t tokens, std::size_t offsets[4]);

void moe_slots_launch(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                      const MoeSlotsLayer& layer, const MoeSlotPool& pool, MoeSlotsOutput mode,
                      Tensor& out, const MoeSlotsWorkspace& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
