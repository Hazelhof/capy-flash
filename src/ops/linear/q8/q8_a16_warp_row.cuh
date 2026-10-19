#pragma once

// Q8G32 RowSplit decode contraction with register-resident row slices (T <= 4).
//
// Same arithmetic as q8_a16_simt_kernel: one or more warps per output row, split at scale-pair
// boundaries; lane l owns bytes [8l, 8l + 8) of every 256-wide K phase of its warp's slice,
// dequantizes them as fp32(int8) * fp32(fp16 scale) and accumulates one fmaf per element in
// ascending K order; the warp reduces with the same shuffle tree and a row's split warps add their
// partials in split order. For K % 64 == 0 the outputs are therefore bit-identical to the SIMT
// kernel with the same warps per row. Only the memory pipeline differs: every code and scale load
// of the slice is issued into registers before the first use (no shared staging and no cp.async
// stages), so each warp keeps its whole slice in flight and occupancy is bounded by registers.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q8/q8_schedule.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ uint2 q8_warp_row_load_codes(const std::uint8_t* ptr) {
    uint2 value;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(value.x), "=r"(value.y)
                 : "l"(ptr));
    return value;
}

__device__ __forceinline__ unsigned short q8_warp_row_load_scale(const std::uint8_t* ptr) {
    unsigned short value;
    asm volatile("ld.global.nc.u16 %0, [%1];\n" : "=h"(value) : "l"(ptr));
    return value;
}

// Tokens = the exact activation columns. FullSlice: every warp slice fills all phases.
template <class Schedule, int Tokens, bool FullSlice, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q8_a16_warp_row_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, Epilogue epilogue, int rows, int k,
    int padded_k) {
    constexpr int R = Schedule::kBlockRows;
    constexpr int W = Schedule::kWarpsPerRow;
    constexpr int T = Tokens;
    constexpr int P = Schedule::kPhases;
    static_assert(T >= 1 && T <= Schedule::kTokenCapacity);
    __shared__ float partial[W > 1 ? R : 1][W][T];

    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int local_row = warp / W;
    const int split     = warp % W;
    const int row       = static_cast<int>(blockIdx.x) * R + local_row;
    if constexpr (W == 1) {
        if (row >= rows) return;
    }
    const bool live_row = row < rows;
    const int safe_row  = live_row ? row : 0;
    // Slices end on scale pairs, as in the SIMT kernel (the launcher requires K % 64 == 0).
    const int pairs       = k / 64;
    const int group_begin = (pairs * split / W) * 2;
    const int groups      = live_row ? (pairs * (split + 1) / W) * 2 - group_begin : 0;

    const std::int64_t group0 = static_cast<std::int64_t>(safe_row) * (padded_k / 32) + group_begin;
    const std::uint8_t* code_row  = codes + group0 * 32 + lane * 8;
    const std::uint8_t* scale_row = scales + (group0 + (lane >> 2)) * 2;
    const __nv_bfloat16* lane_x   = x + group_begin * 32 + lane * 8;

    // Unconditional loads (lanes past a partial slice re-read its first phase) keep every load of
    // the slice ahead of the first use; the contraction skips those lanes.
    uint2 packed[P];
    unsigned short scale_bits[P];
#pragma unroll
    for (int phase = 0; phase < P; ++phase) {
        const int offset  = FullSlice || phase * 8 + (lane >> 2) < groups ? phase : 0;
        packed[phase]     = q8_warp_row_load_codes(code_row + offset * 256);
        scale_bits[phase] = q8_warp_row_load_scale(scale_row + offset * 16);
    }

    float acc[T] = {};
#pragma unroll
    for (int phase = 0; phase < P; ++phase) {
        if (FullSlice || phase * 8 + (lane >> 2) < groups) {
            const float scale = __half2float(__ushort_as_half(scale_bits[phase]));
            float weights[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                weights[i] = static_cast<float>(static_cast<std::int8_t>(
                                 (i < 4 ? packed[phase].x : packed[phase].y) >> ((i & 3) * 8))) *
                             scale;
#pragma unroll
            for (int col = 0; col < T; ++col) {
                const uint4 values =
                    load_vec<uint4>(lane_x + static_cast<std::int64_t>(col) * k + phase * 256);
                const float2 x0 = bf16x2_bits_to_float2(values.x);
                const float2 x1 = bf16x2_bits_to_float2(values.y);
                const float2 x2 = bf16x2_bits_to_float2(values.z);
                const float2 x3 = bf16x2_bits_to_float2(values.w);
                acc[col]        = fmaf(weights[0], x0.x, acc[col]);
                acc[col]        = fmaf(weights[1], x0.y, acc[col]);
                acc[col]        = fmaf(weights[2], x1.x, acc[col]);
                acc[col]        = fmaf(weights[3], x1.y, acc[col]);
                acc[col]        = fmaf(weights[4], x2.x, acc[col]);
                acc[col]        = fmaf(weights[5], x2.y, acc[col]);
                acc[col]        = fmaf(weights[6], x3.x, acc[col]);
                acc[col]        = fmaf(weights[7], x3.y, acc[col]);
            }
        }
    }

    float sums[T];
#pragma unroll
    for (int token = 0; token < T; ++token) {
        sums[token] = warp_reduce_sum(acc[token]);
        if constexpr (W > 1) {
            if (lane == 0) partial[local_row][split][token] = sums[token];
        }
    }
    if constexpr (W > 1) {
        __syncthreads();
        if (split == 0 && lane == 0 && live_row) {
#pragma unroll
            for (int token = 0; token < T; ++token) {
                sums[token] = partial[local_row][0][token];
#pragma unroll
                for (int part = 1; part < W; ++part) sums[token] += partial[local_row][part][token];
            }
            linear_finish_row(output, epilogue, row, 0, sums, T);
        }
    } else if (lane == 0) {
        linear_finish_row(output, epilogue, row, 0, sums, T);
    }
}

} // namespace ninfer::ops::detail
