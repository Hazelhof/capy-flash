#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Qwen4Exp per-layer n-gram embedding (PLE) injection. One PLE block computes, per column t,
// with E the 2560-wide n-gram embedding (16 decoded table rows) and R the wide residual:
//
//   key_value  = Q8 linear([key_proj ; value_proj], E)             -> [key (S*H) ; value (H)]
//   u          = ple_gate(R, key_value, query_norm, key_norm)     -> [S*H]
//   c          = grouped_offset_rmsnorm(u, conv_norm, S)          (hyper_connection.h)
//   R         += u + silu(depthwise_conv(c; kernel K, dilation d)) (ple_dilated_conv)
//   history   <- last (K-1)*d columns of [history ; c]             (ple_conv_advance)
//
// The Ops below own the gate and the convolution transition; the projection is ops::linear and
// the table gather is host work of the model.

/**
 * Op: PLE stream gate with fused grouped offset RMSNorms
 *
 * Math / indexing (per column t and stream s; key[i] = key_value[i], value[h] = key_value[S*H+h]):
 *   q[s*H+h]  = residual[s*H+h] / rms_s(residual) * (1 + query_norm[s*H+h])
 *   k[s*H+h]  = key[s*H+h]      / rms_s(key)      * (1 + key_norm[s*H+h])
 *     with rms_s(x) = sqrt((1/H) sum_h x[s*H+h]^2 + eps)
 *   g[s]      = sum_h k[s*H+h] * q[s*H+h] / sqrt(H)
 *   g'[s]     = sign(g) * sqrt(max(|g|, 1e-6))
 *   gated[s*H+h] = sigmoid(g'[s]) * value[h]
 *
 * Logical shapes: residual, gated [S*H, T]; key_value [(S+1)*H, T] (the output of the grouped
 *   key/value projection); query_norm, key_norm [S*H].
 * Supported domain: BF16 contiguous; 1 <= S <= 16; H >= 1; T >= 1; eps > 0 and finite.
 * Numeric: FP32 reductions; one BF16 rounding of `gated`. FP64 oracle from represented inputs.
 * Effects: writes `gated`; must not overlap any input.  Workspace: none.
 * Execution: stream ordered, CUDA Graph capturable.
 */
void ple_gate(const Tensor& residual, const Tensor& key_value, const Tensor& query_norm,
              const Tensor& key_norm, std::int32_t streams, float eps, Tensor& gated,
              cudaStream_t stream);

/**
 * Op: causal dilated depthwise convolution with SiLU, added to the residual with the gated values
 *
 * Math / indexing (lane b, valid column w, channel c; x = [history(source_slots[b]) ; normed]):
 *   conv[c,w,b]      = sum_{j<K} weight[j,c] * x[w - (K-1-j)*dilation]
 *   residual[c,w,b] += gated[c,w,b] + silu(conv[c,w,b])
 *
 * Logical shapes: normed, gated, residual [C, W, B]; weight [K, C] stored ne = {C, K} (tap major,
 *   tap K-1 multiplies the current column); states [(K-1)*dilation*C, Slots], slot s holding the
 *   (K-1)*dilation previous `normed` columns oldest first; source_slots I32 [B].
 * Supported domain: BF16 contiguous; K >= 2; 1 <= (K-1)*dilation <= 32.
 * Numeric: FP32 convolution and sum, one BF16 rounding of each residual element.
 * Effects: updates `residual` in place; `states` is read only (ple_conv_advance commits).
 * Workspace: none.  Execution: stream ordered, CUDA Graph capturable.
 */
void ple_dilated_conv(const Tensor& normed, const Tensor& gated, const Tensor& weight,
                      std::int32_t dilation, const Tensor& states, const Tensor& source_slots,
                      Tensor& residual, cudaStream_t stream);

/**
 * Op: advance the PLE convolution history
 *
 * Math / indexing:
 *   slot destination_slots[b] receives the last `history` columns of
 *   [states(source_slots[b]) ; normed[:, 0:count_b, b]], count_b = min(W, max(0, valid[b])) or W
 *   when `valid_columns` is null.
 *
 * Logical shapes: normed [C, W, B]; states [history*C, Slots]; slot tables and valid I32 [B].
 * Supported domain: BF16 states/normed; 1 <= history <= 32. A lane's source slot may equal any
 *   lane's destination slot: every source is read before any destination is written.
 * Numeric: exact copies.  Effects: writes the destination slots only.  Workspace: none.
 * Execution: stream ordered, CUDA Graph capturable.
 */
void ple_conv_advance(const Tensor& normed, const Tensor* valid_columns, std::int32_t history,
                      Tensor& states, const Tensor& source_slots, const Tensor& destination_slots,
                      cudaStream_t stream);

} // namespace ninfer::ops
