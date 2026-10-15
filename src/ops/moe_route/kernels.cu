// Implements: include/ninfer/ops/moe_route.h
// The routing kernels are ninfer-ext's moe_route (giveen/ninfer-ext @9c933f01,
// src/ops/offload_moe/kernels.cu, Apache-2.0): one warp per router row keeps its row in registers
// and sweeps a column tile, so the 2.6 MB router streams through many SMs even for one token;
// the top-k runs one warp per column.
#include "ops/moe_route/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int H             = kMoeRouteHidden;
constexpr int E             = kMoeRouteExperts;
constexpr int K             = kMoeRouteTopK;
constexpr int kRouteWarps   = 4;
constexpr int kRouteColumns = 64;
constexpr int kRouteVectors = H / 8 / 32; // uint4 (8 x BF16) per lane

__device__ __forceinline__ float dot_bf16x8(uint4 a, uint4 b) {
    const auto* x = reinterpret_cast<const __nv_bfloat162*>(&a);
    const auto* y = reinterpret_cast<const __nv_bfloat162*>(&b);
    float sum     = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 u = __bfloat1622float2(x[i]);
        const float2 v = __bfloat1622float2(y[i]);
        sum            = fmaf(u.x, v.x, fmaf(u.y, v.y, sum));
    }
    return sum;
}

__global__ void __launch_bounds__(kRouteWarps * 32)
    moe_route_logits_kernel(const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ router, float* __restrict__ logits,
                            std::int32_t columns) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * kRouteWarps + warp;
    if (row > E) { return; }
    const auto* w4 = reinterpret_cast<const uint4*>(router + static_cast<std::int64_t>(row) * H);
    uint4 w[kRouteVectors];
#pragma unroll
    for (int i = 0; i < kRouteVectors; ++i) { w[i] = __ldg(w4 + i * 32 + lane); }
    // PDL (T49): the router rows above do not depend on the previous kernel, so they stream in
    // while it drains; x is read only after the wait. The top-k kernel waits at its entry.
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int begin = static_cast<int>(blockIdx.y) * kRouteColumns;
    const int end   = min(columns, begin + kRouteColumns);
    for (int col = begin; col < end; ++col) {
        const auto* x4 = reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(col) * H);
        float sum      = 0.0F;
#pragma unroll
        for (int i = 0; i < kRouteVectors; ++i) { sum += dot_bf16x8(w[i], x4[i * 32 + lane]); }
        sum = warp_sum(sum);
        if (lane == 0) { logits[static_cast<std::int64_t>(col) * (E + 1) + row] = sum; }
    }
}

__global__ void __launch_bounds__(256)
    moe_route_topk_kernel(const float* __restrict__ logits, std::int32_t columns,
                          std::int32_t* __restrict__ ids, float* __restrict__ weights,
                          float* __restrict__ shared_gate, const std::int32_t* __restrict__ valid,
                          std::int32_t width, bool last_valid_only) {
    constexpr int kPerLane = E / 32;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (col >= columns) { return; }
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const float* l = logits + col * (E + 1);
    float v[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) { v[i] = l[i * 32 + lane]; }
    unsigned taken = 0;
    float top[K];
    int top_id[K];
    for (int k = 0; k < K; ++k) {
        float best  = -CUDART_INF_F;
        int best_id = E;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int e = i * 32 + lane;
            if (!((taken >> i) & 1U) && (v[i] > best || (v[i] == best && e < best_id))) {
                best    = v[i];
                best_id = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_v = __shfl_xor_sync(0xffffffffu, best, offset);
            const int other_id  = __shfl_xor_sync(0xffffffffu, best_id, offset);
            if (other_v > best || (other_v == best && other_id < best_id)) {
                best    = other_v;
                best_id = other_id;
            }
        }
        top[k]    = best;
        top_id[k] = best_id;
        if ((best_id & 31) == lane) { taken |= 1U << (best_id >> 5); }
    }
    if (lane == 0) {
        const bool routed =
            valid == nullptr || (last_valid_only ? col % width == valid[col / width] - 1
                                                 : col % width < valid[col / width]);
        float total = 0.0F;
        for (int k = 0; k < K; ++k) { total += __expf(top[k] - top[0]); }
        for (int k = 0; k < K; ++k) {
            ids[col * K + k]     = routed ? top_id[k] : -1;
            weights[col * K + k] = routed ? __expf(top[k] - top[0]) / total : 0.0F;
        }
        shared_gate[col] = sigmoid(l[E]);
    }
}

__global__ void moe_combine_shared_kernel(const float* __restrict__ partial,
                                          const float* __restrict__ shared_gate,
                                          const __nv_bfloat16* __restrict__ shared,
                                          __nv_bfloat16* __restrict__ y, std::int32_t rows,
                                          std::int64_t count) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t t = i / rows;
        y[i] = __float2bfloat16_rn(fmaf(shared_gate[t], __bfloat162float(shared[i]), partial[i]));
    }
}

__global__ void moe_accumulate_shared_kernel(const float* __restrict__ shared_gate,
                                             const __nv_bfloat16* __restrict__ shared,
                                             float* __restrict__ partial, std::int32_t rows,
                                             std::int64_t count) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t t = i / rows;
        partial[i] = fmaf(shared_gate[t], __bfloat162float(shared[i]), partial[i]);
    }
}

} // namespace

void moe_route_launch(const Tensor& x, const Tensor& router, float* logits, Tensor& ids,
                      Tensor& weights, Tensor& shared_gate, const Tensor* valid_columns,
                      bool last_valid_only, cudaStream_t stream) {
    const int columns = x.ne[1];
    const int width   = valid_columns != nullptr ? columns / valid_columns->ne[0] : columns;
    const dim3 grid((E + 1 + kRouteWarps - 1) / kRouteWarps,
                    (columns + kRouteColumns - 1) / kRouteColumns);
    // Both launches are programmatic dependents of their predecessor (T49): each kernel waits
    // before it reads anything a previous kernel wrote.
    CUDA_CHECK(pdl::launch_dependent({grid, dim3(kRouteWarps * 32), 0, stream},
                                     moe_route_logits_kernel,
                                     static_cast<const __nv_bfloat16*>(x.data),
                                     static_cast<const __nv_bfloat16*>(router.data), logits,
                                     static_cast<std::int32_t>(columns)));
    CUDA_CHECK(pdl::launch_dependent(
        {dim3((columns + 7) / 8), dim3(256), 0, stream}, moe_route_topk_kernel,
        static_cast<const float*>(logits), static_cast<std::int32_t>(columns),
        static_cast<std::int32_t*>(ids.data), static_cast<float*>(weights.data),
        static_cast<float*>(shared_gate.data),
        valid_columns != nullptr ? static_cast<const std::int32_t*>(valid_columns->data) : nullptr,
        static_cast<std::int32_t>(width), last_valid_only));
}

void moe_combine_shared_launch(const Tensor& partial, const Tensor& shared_gate,
                               const Tensor& shared, Tensor& y, cudaStream_t stream) {
    const std::int64_t count = partial.numel();
    const int threads        = 256;
    const auto blocks = static_cast<int>(std::min<std::int64_t>((count + threads - 1) / threads,
                                                                std::int64_t(70) * 16));
    moe_combine_shared_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(partial.data), static_cast<const float*>(shared_gate.data),
        static_cast<const __nv_bfloat16*>(shared.data), static_cast<__nv_bfloat16*>(y.data),
        partial.ne[0], count);
    CUDA_CHECK(cudaGetLastError());
}

void moe_accumulate_shared_launch(const Tensor& shared_gate, const Tensor& shared, Tensor& partial,
                                  cudaStream_t stream) {
    const std::int64_t count = partial.numel();
    const int threads        = 256;
    const auto blocks = static_cast<int>(std::min<std::int64_t>((count + threads - 1) / threads,
                                                                std::int64_t(70) * 16));
    moe_accumulate_shared_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(shared_gate.data),
        static_cast<const __nv_bfloat16*>(shared.data), static_cast<float*>(partial.data),
        partial.ne[0], count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
