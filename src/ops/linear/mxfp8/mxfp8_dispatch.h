#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// MXFP8_E4M3FN_G32_UE8M0 / BlockScaleK32, A16 only (BF16 activations, no input scale). Writes BF16
// or FP32 out (FP32: row-parallel partials exchanged before the single BF16 rounding).
std::size_t mxfp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens);

void mxfp8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                    cudaStream_t stream);

} // namespace ninfer::ops::detail
