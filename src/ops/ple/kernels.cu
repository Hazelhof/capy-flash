// Implements: include/ninfer/ops/ple.h
// Column-parallel SIMT kernels. Hashing is exact integer arithmetic; every floating-point map
// accumulates in FP32 and rounds once to BF16.
#include "ops/ple/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock          = 256;
constexpr int kMaximumHistory = 32;

int grid_for(std::int64_t work) {
    return static_cast<int>(std::clamp<std::int64_t>((work + kBlock - 1) / kBlock, 1, 1 << 20));
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    value          = warp_sum(value);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
    if (warp == 0) {
        total = lane < static_cast<int>(blockDim.x >> 5) ? scratch[lane] : 0.0F;
        total = warp_sum(total);
        if (lane == 0) { scratch[0] = total; }
    }
    __syncthreads();
    total = scratch[0];
    __syncthreads();
    return total;
}

// One CTA per (stream, column): the two grouped offset RMSNorms are folded into one pass,
//   k.q = inv_k * inv_q * sum_h key*(1+key_norm) * residual*(1+query_norm).
__global__ void ple_gate_kernel(const __nv_bfloat16* __restrict__ residual,
                                const __nv_bfloat16* __restrict__ key_value,
                                const __nv_bfloat16* __restrict__ query_norm,
                                const __nv_bfloat16* __restrict__ key_norm,
                                __nv_bfloat16* __restrict__ gated, std::int32_t width,
                                std::int32_t streams, float eps) {
    __shared__ float scratch[32];
    const std::int32_t s      = static_cast<std::int32_t>(blockIdx.x) % streams;
    const std::int64_t column = static_cast<std::int64_t>(blockIdx.x) / streams;
    const std::int64_t base =
        column * width * streams + static_cast<std::int64_t>(s) * width;
    const std::int64_t norm_base = static_cast<std::int64_t>(s) * width;
    // key_value column: [key (S*H) ; value (H)].
    const __nv_bfloat16* key   = key_value + column * width * (streams + 1) + norm_base;
    const __nv_bfloat16* value = key_value + column * width * (streams + 1) +
                                 static_cast<std::int64_t>(streams) * width;
    float qq = 0.0F, kk = 0.0F, qk = 0.0F;
    for (int h = static_cast<int>(threadIdx.x); h < width; h += blockDim.x) {
        const float q = __bfloat162float(residual[base + h]);
        const float k = __bfloat162float(key[h]);
        qq            = fmaf(q, q, qq);
        kk            = fmaf(k, k, kk);
        qk            = fmaf(q * (1.0F + __bfloat162float(query_norm[norm_base + h])),
                             k * (1.0F + __bfloat162float(key_norm[norm_base + h])), qk);
    }
    qq                 = block_sum(qq, scratch);
    kk                 = block_sum(kk, scratch);
    qk                 = block_sum(qk, scratch);
    const float inv_h  = 1.0F / static_cast<float>(width);
    const float g      = qk * rsqrtf(qq * inv_h + eps) * rsqrtf(kk * inv_h + eps) *
                    rsqrtf(static_cast<float>(width));
    const float root   = sqrtf(fmaxf(fabsf(g), 1e-6F));
    const float sign   = g > 0.0F ? 1.0F : (g < 0.0F ? -1.0F : 0.0F);
    const float gate   = sigmoid(sign * root);
    for (int h = static_cast<int>(threadIdx.x); h < width; h += blockDim.x) {
        gated[base + h] = __float2bfloat16_rn(gate * __bfloat162float(value[h]));
    }
}

__global__ void ple_dilated_conv_kernel(const __nv_bfloat16* __restrict__ normed,
                                        const __nv_bfloat16* __restrict__ gated,
                                        const __nv_bfloat16* __restrict__ weight,
                                        const __nv_bfloat16* __restrict__ states,
                                        const std::int32_t* __restrict__ source_slots,
                                        __nv_bfloat16* __restrict__ residual, std::int32_t channels,
                                        std::int32_t width, std::int32_t lanes, std::int32_t taps,
                                        std::int32_t dilation) {
    const std::int64_t total   = static_cast<std::int64_t>(channels) * width * lanes;
    const std::int32_t history = (taps - 1) * dilation;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t c    = static_cast<std::int32_t>(i % channels);
        const std::int64_t wb   = i / channels;
        const std::int32_t w    = static_cast<std::int32_t>(wb % width);
        const std::int32_t lane = static_cast<std::int32_t>(wb / width);
        const std::int64_t slot = source_slots[lane];
        float sum               = 0.0F;
        for (int j = 0; j < taps; ++j) {
            const std::int32_t m = w - (taps - 1 - j) * dilation;
            float x;
            if (m >= 0) {
                x = __bfloat162float(
                    normed[(static_cast<std::int64_t>(lane) * width + m) * channels + c]);
            } else {
                const std::int32_t row = history + m;
                x                      = __bfloat162float(states[slot * history * channels +
                                            static_cast<std::int64_t>(row) * channels + c]);
            }
            sum += __bfloat162float(weight[static_cast<std::int64_t>(j) * channels + c]) * x;
        }
        residual[i] = __float2bfloat16_rn(__bfloat162float(residual[i]) +
                                          __bfloat162float(gated[i]) + silu(sum));
    }
}

// One thread per (lane, channel). The block spans every lane for its channel tile, so a lane's
// source rows are snapshotted into registers before the barrier: reads never observe another
// lane's destination write (a lane's source slot may equal another lane's destination slot).
__global__ void ple_conv_advance_kernel(const __nv_bfloat16* __restrict__ normed,
                                        const std::int32_t* __restrict__ valid_columns,
                                        __nv_bfloat16* __restrict__ states,
                                        const std::int32_t* __restrict__ source_slots,
                                        const std::int32_t* __restrict__ destination_slots,
                                        std::int32_t channels, std::int32_t width,
                                        std::int32_t lanes, std::int32_t tile,
                                        std::int32_t history) {
    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x) / tile;
    const std::int32_t c    = static_cast<std::int32_t>(blockIdx.x) * tile +
                              static_cast<std::int32_t>(threadIdx.x) % tile;
    __nv_bfloat16 values[kMaximumHistory];
    if (c < channels) {
        std::int32_t count = width;
        if (valid_columns != nullptr) { count = max(0, min(width, valid_columns[lane])); }
        const std::int64_t source = source_slots[lane];
        for (std::int32_t r = 0; r < history; ++r) {
            const std::int32_t index = count + r;
            values[r] =
                index < history
                    ? states[(source * history + index) * channels + c]
                    : normed[(static_cast<std::int64_t>(lane) * width + (index - history)) *
                                 channels +
                             c];
        }
    }
    __syncthreads();
    if (c < channels) {
        const std::int64_t destination = destination_slots[lane];
        for (std::int32_t r = 0; r < history; ++r) {
            states[(destination * history + r) * channels + c] = values[r];
        }
    }
}

} // namespace

void ple_gate_launch(const Tensor& residual, const Tensor& key_value, const Tensor& query_norm,
                     const Tensor& key_norm, std::int32_t streams, float eps, Tensor& gated,
                     cudaStream_t stream) {
    const std::int32_t width   = key_value.ne[0] / (streams + 1);
    const std::int64_t columns = key_value.ne[1];
    ple_gate_kernel<<<static_cast<unsigned>(columns * streams), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(key_value.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const __nv_bfloat16*>(key_norm.data), static_cast<__nv_bfloat16*>(gated.data),
        width, streams, eps);
    CUDA_CHECK(cudaGetLastError());
}

void ple_dilated_conv_launch(const Tensor& normed, const Tensor& gated, const Tensor& weight,
                             std::int32_t dilation, const Tensor& states,
                             const Tensor& source_slots, Tensor& residual, cudaStream_t stream) {
    ple_dilated_conv_kernel<<<grid_for(normed.numel()), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normed.data),
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(states.data),
        static_cast<const std::int32_t*>(source_slots.data),
        static_cast<__nv_bfloat16*>(residual.data), normed.ne[0], normed.ne[1], normed.ne[2],
        weight.ne[1], dilation);
    CUDA_CHECK(cudaGetLastError());
}

void ple_conv_advance_launch(const Tensor& normed, const Tensor* valid_columns,
                             std::int32_t history, Tensor& states, const Tensor& source_slots,
                             const Tensor& destination_slots, cudaStream_t stream) {
    const std::int32_t channels = normed.ne[0];
    const std::int32_t lanes    = normed.ne[2];
    // One block spans every lane for `tile` channels so the register snapshot sees the whole
    // pre-update history, whatever slot permutation the lane table encodes.
    const std::int32_t tile = std::max(1, kBlock / std::max(1, lanes));
    const int grid          = static_cast<int>((channels + tile - 1) / tile);
    ple_conv_advance_kernel<<<grid, tile * lanes, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normed.data),
        valid_columns != nullptr ? static_cast<const std::int32_t*>(valid_columns->data) : nullptr,
        static_cast<__nv_bfloat16*>(states.data),
        static_cast<const std::int32_t*>(source_slots.data),
        static_cast<const std::int32_t*>(destination_slots.data), channels, normed.ne[1], lanes,
        tile, history);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
