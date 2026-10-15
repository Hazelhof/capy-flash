#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Router-ahead prediction of the routed experts of the next Qwen4Exp (Qwen3.8-Flash-Next) MoE
// layer, for expert prefetch (predictor `post_uni` of T19, prefetch T25; see docs/capy-flash/development-log.md).

inline constexpr std::int32_t kMoePredictHidden        = 2560;
inline constexpr std::int32_t kMoePredictStreams       = 4;
inline constexpr std::int32_t kMoePredictExperts       = 512;
inline constexpr std::int32_t kMoePredictMaxTokens     = 8;
inline constexpr std::int32_t kMoePredictMaxCandidates = 32;

/**
 * Op: router-ahead window scores of the next MoE layer and its top non-resident candidates
 *
 * Math / indexing, per column t of the wide residual x [S*H, T] of layer l after its mixer output
 * was committed (the input of layer l's MLP hyper-connection), with layer j = l+1's MLP
 * hyper-connection norm weight `norm` [S*H] and router `router` (the 512 routed rows):
 *   inv[s,t]  = 1 / sqrt((1/H) sum_h x[s*H+h,t]^2 + eps)
 *   n[s,h,t]  = BF16(x[s*H+h,t] * inv[s,t] * (1 + norm[s*H+h]))     (grouped offset RMSNorm)
 *   u[h,t]    = BF16((1/S) sum_s n[s,h,t])                           (stream mean, no dynamic mix)
 *   l[e,t]    = router[e] . u[:,t]                                   (FP32 accumulation)
 *   p[e,t]    = exp(l[e,t] - max_e l[:,t]) / sum_e exp(...)          (softmax over the 512)
 *   score[e]  = sum_t p[e,t]                                         (window score)
 *   eligible  = { e : resident[e] < 0 and score[e] >= min_score }
 *   candidates[k], scores[k] = the k-th eligible expert by descending score (an exact tie goes to
 *               the lower id), for k < count; -1 / 0 past the eligible ones.
 * Then `*flag = 1` after a system-scope fence of every candidate write, so a host that sees the
 * flag (mapped memory) reads complete candidates.
 *
 * Logical shapes: x BF16 [S*H, T] (S = 4, H = 2560, 1 <= T <= 8), norm BF16 [S*H], router BF16
 *   [2560, 513] (the routed rows then the shared-expert row, as moe_route), resident I32 [512]
 *   (a slot table row: >= 0 = resident), candidates I32 [count], scores FP32 [count],
 *   1 <= count <= 32. candidates / scores / flag may be host-mapped memory.
 * Numeric: FP32 reductions; u is rounded to BF16 as the reference (`router_probe.py` uni mode).
 *   A predictor: tolerance is the selection itself (the test checks it against an FP64 oracle up
 *   to near-ties of the score).
 * Effects: writes the candidates, scores and flag. Workspace: moe_predict_workspace_bytes()
 *   (device memory, also a completion counter the last block resets; one call at a time per
 *   workspace). Execution: stream ordered, CUDA Graph capturable.
 */
void moe_predict_ahead(const Tensor& x, const Tensor& norm, const Tensor& router, float eps,
                       const std::int32_t* resident, float min_score, std::int32_t count,
                       std::int32_t* candidates, float* scores, std::uint32_t* flag,
                       void* workspace, cudaStream_t stream);

// Device workspace of one moe_predict_ahead (any T <= 8). Must be zero-initialized once (the
// completion counter); every call leaves it reset.
[[nodiscard]] std::size_t moe_predict_workspace_bytes();

} // namespace ninfer::ops
