#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Router and output combine (or tensor-parallel accumulation) of the Qwen3.8-Flash-Next (Qwen4Exp)
// sparse MoE block: hidden 2560, 512 routed experts, top-10, and one sigmoid-gated shared expert.
// The routed experts themselves are computed elsewhere (moe_slots.h, or a host path), as an FP32
// partial sum.

inline constexpr std::int32_t kMoeRouteHidden  = 2560;
inline constexpr std::int32_t kMoeRouteExperts = 512;
inline constexpr std::int32_t kMoeRouteTopK    = 10;

/**
 * Op: Qwen4Exp router, top-k selection and shared-expert gate
 *
 * Math / indexing, per column t (x = x[:,t]):
 *   l_e         = router[e] . x                      for e < 512 (FP32 accumulation)
 *   s           = router[512] . x                    (shared-expert score row)
 *   ids[:,t]    = the 10 experts with the largest l_e, in decreasing order; an exact tie goes to
 *                 the lower expert id
 *   weights[k,t]= exp(l_{ids[k]} - l_max) / sum_j exp(l_{ids[j]} - l_max)
 *                 (= softmax over all 512 experts, top-10, renormalized: the denominators cancel)
 *   shared_gate[t] = sigmoid(s)
 *
 *   Columns past their lane's valid count (`valid_columns`, I32 [B] with B dividing T: column
 *   t = b*W + w, W = T/B, is valid iff w < valid_columns[b]) route nothing: ids[:,t] = -1 and
 *   weights[:,t] = 0, which every routed-expert backend skips (the speculative verify and MTP
 *   columns past a lane's extent, whose outputs are never committed); shared_gate[t] is written.
 *   With `routed_columns` = MoeRouteColumns::LastValid only the lane's last valid column
 *   (w == valid_columns[b] - 1) routes and every other column routes nothing (the MTP alignment
 *   of a verified window, whose output is read only at the accepted column). A lane whose valid
 *   count is 0 routes nothing in either mode.
 *
 * Logical shapes: x BF16 [2560,T]; router BF16 [2560,513] (ne order: 513 rows of 2560, the 512
 *   expert rows then the shared score row, contiguous); ids I32 [10,T]; weights FP32 [10,T];
 *   shared_gate FP32 [T]; valid_columns null or I32 [B] (device; required by LastValid).
 * Supported domain: T >= 1, contiguous, x and router 16-byte aligned, finite inputs.
 * Numeric: FP32 dot products over represented BF16 inputs (no BF16 rounding of the logits);
 *   FP32 softmax. The oracle evaluates the logits in FP64; the selection may differ from the
 *   FP64 selection only between experts whose FP64 logits are within the FP32 accumulation bound
 *   of each other.
 * Effects: writes ids, weights and shared_gate. Workspace: moe_route_workspace_bytes(T) (the
 *   FP32 logits). Execution: stream ordered, CUDA Graph capturable.
 *
 * The kernels are ninfer-ext's `moe_route` (giveen/ninfer-ext @9c933f01, Apache-2.0).
 */
enum class MoeRouteColumns : std::uint8_t {
    ValidPrefix, // the columns w < valid_columns[b] route
    LastValid,   // only the column w == valid_columns[b] - 1 routes
};

void moe_route(const Tensor& x, const Tensor& router, WorkspaceArena& workspace, Tensor& ids,
               Tensor& weights, Tensor& shared_gate, cudaStream_t stream,
               const Tensor* valid_columns    = nullptr,
               MoeRouteColumns routed_columns = MoeRouteColumns::ValidPrefix);

[[nodiscard]] std::size_t moe_route_workspace_bytes(std::int32_t tokens);

/**
 * Op: MoE output = routed partial + gated shared expert, one rounding
 *
 *   y[h,t] = BF16_RNE( fma(shared_gate[t], shared[h,t], partial[h,t]) )   (FP32 fma)
 *
 * `partial` is the FP32 routed-expert sum (every routed contribution, from any backend, already
 * accumulated in FP32), so the block output is rounded to BF16 exactly once.
 * Logical shapes: partial FP32 [H,T]; shared_gate FP32 [T]; shared BF16 [H,T]; y BF16 [H,T];
 *   H >= 1, T >= 1, contiguous. y must not overlap the inputs.
 * Numeric: exact (one FP32 fma, one RNE rounding). Workspace: none. Graph capturable.
 */
void moe_combine_shared(const Tensor& partial, const Tensor& shared_gate, const Tensor& shared,
                        Tensor& y, cudaStream_t stream);

/**
 * Op: gated shared expert accumulated into the FP32 partial, no rounding
 *
 *   partial[h,t] = fma(shared_gate[t], shared[h,t], partial[h,t])          (FP32 fma, in place)
 *
 * The tensor-parallel form of moe_combine_shared: at tp 2 the shared expert of a block is held by
 * one rank, which adds it to its FP32 partial before the cross-rank FP32 all-reduce
 * (allreduce_sum_f32), so the block output is still rounded to BF16 once, after the sum.
 * BF16_RNE(moe_accumulate_shared(p)) equals moe_combine_shared(p) bit for bit.
 * Logical shapes: shared_gate FP32 [T]; shared BF16 [H,T]; partial FP32 [H,T]; H >= 1, T >= 1,
 *   contiguous; partial must not overlap the inputs.
 * Numeric: exact (one FP32 fma per element). Workspace: none. Graph capturable.
 */
void moe_accumulate_shared(const Tensor& shared_gate, const Tensor& shared, Tensor& partial,
                           cudaStream_t stream);

} // namespace ninfer::ops
