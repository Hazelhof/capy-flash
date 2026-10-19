// Implements: include/ninfer/ops/moe_predict.h
// Router-ahead window scores for expert prefetch (predictor `post_uni` of T19, prefetch T25;
// docs/capy-flash/development-log.md): a grouped offset RMSNorm + stream mean of the wide residual,
// the next layer's 512 router rows (one warp per row, as moe_route's logits kernel), and in the
// grid's last block the softmax window score and the top non-resident candidates by rank counting
// (deterministic, no sort).
#include "ninfer/ops/moe_predict.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int H            = kMoePredictHidden;
constexpr int S            = kMoePredictStreams;
constexpr int E            = kMoePredictExperts;
constexpr int MAXT         = kMoePredictMaxTokens;
constexpr int kNormThreads = 256;
constexpr int kPerThread   = H / kNormThreads; // 10
constexpr int kWarps       = 4;                // router rows per block
constexpr int kBlocks      = E / kWarps;       // 128
constexpr int kVectors     = H / 8 / 32;       // uint4 of 8 BF16 per lane

// Workspace: [0, 256) completion counter, then u BF16 [H, MAXT], then logits FP32 [MAXT, E].
constexpr std::size_t kUOffset      = 256;
constexpr std::size_t kLogitsOffset = kUOffset + std::size_t(H) * MAXT * sizeof(__nv_bfloat16);
constexpr std::size_t kWorkspace    = kLogitsOffset + std::size_t(MAXT) * E * sizeof(float);

__global__ void __launch_bounds__(kNormThreads)
    predict_uni_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ norm,
                       float eps, __nv_bfloat16* __restrict__ u) {
    __shared__ float sums[S][kNormThreads / 32];
    __shared__ float inv[S];
    const int t    = static_cast<int>(blockIdx.x);
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31, warp = tid >> 5;
    const __nv_bfloat16* xc = x + static_cast<std::int64_t>(t) * S * H;
    float v[S][kPerThread];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        float sq = 0.0F;
#pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            v[s][i] = __bfloat162float(xc[s * H + i * kNormThreads + tid]);
            sq      = fmaf(v[s][i], v[s][i], sq);
        }
        sq = warp_sum(sq);
        if (lane == 0) { sums[s][warp] = sq; }
    }
    __syncthreads();
    if (tid < S) {
        float total = 0.0F;
        for (int w = 0; w < kNormThreads / 32; ++w) { total += sums[tid][w]; }
        inv[tid] = rsqrtf(total / static_cast<float>(H) + eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
        const int h = i * kNormThreads + tid;
        float mean  = 0.0F;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float n = __bfloat162float(
                __float2bfloat16_rn(v[s][i] * inv[s] * (1.0F + __bfloat162float(norm[s * H + h]))));
            mean += n;
        }
        u[static_cast<std::int64_t>(t) * H + h] = __float2bfloat16_rn(mean / static_cast<float>(S));
    }
}

__device__ __forceinline__ float dot_bf16x8(uint4 a, uint4 b) {
    const auto* x = reinterpret_cast<const __nv_bfloat162*>(&a);
    const auto* y = reinterpret_cast<const __nv_bfloat162*>(&b);
    float sum     = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 p = __bfloat1622float2(x[i]);
        const float2 q = __bfloat1622float2(y[i]);
        sum            = fmaf(p.x, q.x, fmaf(p.y, q.y, sum));
    }
    return sum;
}

__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = v; }
    __syncthreads();
    float m = red[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) { m = fmaxf(m, red[w]); }
    __syncthreads();
    return m;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = v; }
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) { total += red[w]; }
    __syncthreads();
    return total;
}

__global__ void __launch_bounds__(kWarps * 32)
    predict_select_kernel(const __nv_bfloat16* __restrict__ u,
                          const __nv_bfloat16* __restrict__ router, int T,
                          const std::int32_t* __restrict__ resident, float min_score, int count,
                          float* logits, unsigned* counter, std::int32_t* candidates, float* scores,
                          std::uint32_t* flag) {
    __shared__ float s_score[E];
    __shared__ float red[kWarps];
    __shared__ int s_last;
    __shared__ int s_eligible;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31;
    {
        const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
        const auto* w4 =
            reinterpret_cast<const uint4*>(router + static_cast<std::int64_t>(row) * H);
        uint4 w[kVectors];
#pragma unroll
        for (int i = 0; i < kVectors; ++i) { w[i] = __ldg(w4 + i * 32 + lane); }
        for (int t = 0; t < T; ++t) {
            const auto* u4 = reinterpret_cast<const uint4*>(u + static_cast<std::int64_t>(t) * H);
            float sum      = 0.0F;
#pragma unroll
            for (int i = 0; i < kVectors; ++i) { sum += dot_bf16x8(w[i], u4[i * 32 + lane]); }
            sum = warp_sum(sum);
            if (lane == 0) { logits[t * E + row] = sum; }
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) { s_last = atomicAdd(counter, 1U) == gridDim.x - 1; }
    __syncthreads();
    if (!s_last) { return; }
    __threadfence();

    // Last block: window score over the T columns, eligibility, rank by counting.
    constexpr int kPer = E / (kWarps * 32); // 4 experts per thread
    float score[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) { score[i] = 0.0F; }
    for (int t = 0; t < T; ++t) {
        float l[kPer];
        float m = -CUDART_INF_F;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            l[i] = __ldcg(logits + t * E + i * kWarps * 32 + tid);
            m    = fmaxf(m, l[i]);
        }
        m       = block_max(m, red);
        float z = 0.0F;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            l[i] = expf(l[i] - m);
            z += l[i];
        }
        z = block_sum(z, red);
#pragma unroll
        for (int i = 0; i < kPer; ++i) { score[i] += l[i] / z; }
    }
    int eligible = 0;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int e   = i * kWarps * 32 + tid;
        const bool ok = resident[e] < 0 && score[i] >= min_score;
        s_score[e]    = ok ? score[i] : -1.0F; // scores are >= 0: an ineligible one never outranks
        eligible += ok ? 1 : 0;
    }
    if (tid == 0) { s_eligible = 0; }
    __syncthreads();
    atomicAdd(&s_eligible, eligible);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int e   = i * kWarps * 32 + tid;
        const float s = s_score[e];
        if (s < 0.0F) { continue; }
        int rank = 0;
        for (int f = 0; f < E; ++f) {
            const float o = s_score[f];
            rank += (o > s || (o == s && f < e)) ? 1 : 0;
        }
        if (rank < count) {
            candidates[rank] = e;
            scores[rank]     = s;
        }
    }
    __syncthreads();
    for (int k = s_eligible + tid; k < count; k += kWarps * 32) {
        candidates[k] = -1;
        scores[k]     = 0.0F;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *counter                                         = 0;
        *reinterpret_cast<volatile std::uint32_t*>(flag) = 1;
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("moe_predict_ahead: " + message); }
}

} // namespace

std::size_t moe_predict_workspace_bytes() { return kWorkspace; }

void moe_predict_ahead(const Tensor& x, const Tensor& norm, const Tensor& router, float eps,
                       const std::int32_t* resident, float min_score, std::int32_t count,
                       std::int32_t* candidates, float* scores, std::uint32_t* flag,
                       void* workspace, cudaStream_t stream) {
    require(x.dtype == DType::BF16 && x.data != nullptr && x.is_contiguous() && x.ne[0] == S * H &&
                x.ne[1] >= 1 && x.ne[1] <= MAXT && x.ne[2] == 1 && x.ne[3] == 1,
            "x must be BF16 [4*2560, T], 1 <= T <= 8");
    require(norm.dtype == DType::BF16 && norm.data != nullptr && norm.is_contiguous() &&
                norm.numel() == S * H,
            "norm must be BF16 [4*2560]");
    require(router.dtype == DType::BF16 && router.data != nullptr && router.is_contiguous() &&
                router.numel() == static_cast<std::int64_t>(E + 1) * H,
            "router must be BF16 [2560,513]");
    require(reinterpret_cast<std::uintptr_t>(router.data) % 16 == 0,
            "router must be 16-byte aligned");
    require(count >= 1 && count <= kMoePredictMaxCandidates, "count must be 1..32");
    require(resident != nullptr && candidates != nullptr && scores != nullptr && flag != nullptr &&
                workspace != nullptr,
            "null pointer");
    require(reinterpret_cast<std::uintptr_t>(workspace) % 256 == 0,
            "workspace must be 256-byte aligned");
    require(min_score >= 0.0F, "min_score must be >= 0");
    const int T   = x.ne[1];
    auto* base    = static_cast<std::byte*>(workspace);
    auto* counter = reinterpret_cast<unsigned*>(base);
    auto* u       = reinterpret_cast<__nv_bfloat16*>(base + kUOffset);
    auto* logits  = reinterpret_cast<float*>(base + kLogitsOffset);
    predict_uni_kernel<<<T, kNormThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                       static_cast<const __nv_bfloat16*>(norm.data),
                                                       eps, u);
    CUDA_CHECK(cudaGetLastError());
    predict_select_kernel<<<kBlocks, kWarps * 32, 0, stream>>>(
        u, static_cast<const __nv_bfloat16*>(router.data), T, resident, min_score, count, logits,
        counter, candidates, scores, flag);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
