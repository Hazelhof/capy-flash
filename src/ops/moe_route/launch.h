#pragma once

// ninfer::ops::detail - private launch prototypes of moe_route / moe_combine_shared /
// moe_accumulate_shared.

#include "ninfer/ops/moe_route.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void moe_route_launch(const Tensor& x, const Tensor& router, float* logits, Tensor& ids,
                      Tensor& weights, Tensor& shared_gate, const Tensor* valid_columns,
                      bool last_valid_only, cudaStream_t stream);

void moe_combine_shared_launch(const Tensor& partial, const Tensor& shared_gate,
                               const Tensor& shared, Tensor& y, cudaStream_t stream);

void moe_accumulate_shared_launch(const Tensor& shared_gate, const Tensor& shared, Tensor& partial,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
