#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Prepares Gated DeltaNet decay and update gates:
 *
 *   g[h,t]    = -exp(A_log[h]) * softplus(a[h,t] + dt_bias[h])
 *   beta[h,t] = sigmoid(b[h,t]).
 *
 * `a` and `b` are BF16 [H,T] with contiguous heads and any column stride >= H (e.g. row slices of
 * one projection output), `A_log` and `dt_bias` are contiguous FP32 [H], and
 * `g` and `beta` are contiguous FP32 [H,T], with H=48 value heads or H=24 (one rank's half of 48
 * under two-device tensor parallelism; Qwen3.8-Flash-Next). The oracle evaluates the formula
 * naively in FP64; transcendental implementation and intermediate precision are private kernel
 * choices. Inputs and the two outputs must be mutually non-overlapping. There is no workspace or
 * persistent state side effect.
 */
void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops
