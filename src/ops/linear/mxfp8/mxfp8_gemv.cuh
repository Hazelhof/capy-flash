#pragma once

// MXFP8 x BF16 small-T contraction (decode / MTP verification, T <= 8).
//
// Bandwidth-bound: every weight byte is read once. Each warp owns R output rows and a 1/W share
// of their K extent; a lane reads 16 consecutive codes (one 16-byte load, half of a 32-wide
// scale group) per row and step, so a warp step covers 512 contiguous bytes of each row. The
// group dot product is accumulated in FP32 and scaled by the exact power of two once per lane
// and step. Activations are read through the read-only cache (K x T x 2 bytes, L1/L2-resident).

#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/mxfp8/mxfp8_codec.cuh"
#include "ops/linear/mxfp8/mxfp8_operands.h"

#include <stdexcept>

namespace ninfer::ops::detail {

inline constexpr int kMxfp8GemvWarps = 8;

template <int T, int R, int W, class Output>
__global__ __launch_bounds__(kMxfp8GemvWarps *
                             32) void mxfp8_a16_gemv_kernel(Mxfp8LinearOperands op, Output output) {
    static_assert(T >= 1 && T <= 8 && R >= 1 && W >= 1 && kMxfp8GemvWarps % W == 0);
    static_assert(R * T <= 32, "one lane stores one (row, token) result");
    constexpr int kRowsPerBlock = kMxfp8GemvWarps / W * R;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int split             = warp % W;
    const int local_row0        = warp / W * R;
    const int row0              = static_cast<int>(blockIdx.x) * kRowsPerBlock + local_row0;
    const int k                 = op.k;
    const int chunks            = k >> 4;
    const int groups            = k >> 5;

    const std::uint8_t* code_row[R];
    const std::uint8_t* scale_row[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        // Rows past N read row 0 (always valid) and are never stored.
        const int row = row0 + r < op.rows ? row0 + r : 0;
        code_row[r]   = op.codes + static_cast<std::int64_t>(row) * k;
        scale_row[r]  = op.scales + static_cast<std::int64_t>(row) * groups;
    }

    float acc[R][T];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) acc[r][t] = 0.0f;

#pragma unroll 2
    for (int chunk = split * 32 + lane; chunk < chunks; chunk += W * 32) {
        const int kk = chunk << 4;
        uint4 codes[R];
        float scale[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            codes[r] = load_ldg<uint4>(code_row[r] + kk);
            scale[r] = mxfp8_ue8m0_to_float(__ldg(scale_row[r] + (chunk >> 1)));
        }
        float w[R][16];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const unsigned words[4]{codes[r].x, codes[r].y, codes[r].z, codes[r].w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 lo = mxfp8_e4m3x2_to_float2(words[i]);
                const float2 hi = mxfp8_e4m3x2_to_float2(words[i] >> 16);
                w[r][i * 4 + 0] = lo.x;
                w[r][i * 4 + 1] = lo.y;
                w[r][i * 4 + 2] = hi.x;
                w[r][i * 4 + 3] = hi.y;
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const auto* xp = op.x + static_cast<std::int64_t>(t) * k + kk;
            const uint4 xa = load_ldg<uint4>(xp);
            const uint4 xb = load_ldg<uint4>(xp + 8);
            const unsigned xw[8]{xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float2 xv = mxfp8_bf16x2_bits_to_float2(xw[i]);
                    dot             = fmaf(w[r][2 * i], xv.x, dot);
                    dot             = fmaf(w[r][2 * i + 1], xv.y, dot);
                }
                acc[r][t] = fmaf(dot, scale[r], acc[r][t]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) acc[r][t] = warp_sum(acc[r][t]);

    if constexpr (W > 1) {
        __shared__ float partial[kMxfp8GemvWarps][R * T];
        if (lane < R * T) {
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int t = 0; t < T; ++t)
                    if (lane == r * T + t) partial[warp][lane] = acc[r][t];
        }
        __syncthreads();
        if (split == 0 && lane < R * T) {
            const int r   = lane / T;
            const int t   = lane - r * T;
            const int row = row0 + r;
            float sum     = 0.0f;
#pragma unroll
            for (int s = 0; s < W; ++s) sum += partial[warp + s][lane];
            if (row < op.rows) output.store(row, t, sum);
        }
    } else {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < T; ++t)
                if (lane == r * T + t && row0 + r < op.rows) output.store(row0 + r, t, acc[r][t]);
    }
}

// (rows per warp, warps per row) for one launch.
struct Mxfp8GemvShape {
    int rows_per_warp;
    int warps_per_row;
};

template <int T, class Output>
void launch_mxfp8_a16_gemv_t(const Mxfp8LinearOperands& op, Output output, Mxfp8GemvShape shape,
                             cudaStream_t stream) {
    const auto run = [&]<int R, int W>() {
        constexpr int rows_per_block = kMxfp8GemvWarps / W * R;
        mxfp8_a16_gemv_kernel<T, R, W, Output>
            <<<div_up(op.rows, rows_per_block), kMxfp8GemvWarps * 32, 0, stream>>>(op, output);
        CUDA_CHECK(cudaGetLastError());
    };
    if (shape.rows_per_warp == 2 && shape.warps_per_row == 1)
        return run.template operator()<2, 1>();
    if (shape.rows_per_warp == 1 && shape.warps_per_row == 1)
        return run.template operator()<1, 1>();
    if (shape.rows_per_warp == 1 && shape.warps_per_row == 2)
        return run.template operator()<1, 2>();
    if (shape.rows_per_warp == 1 && shape.warps_per_row == 4)
        return run.template operator()<1, 4>();
    if (shape.rows_per_warp == 1 && shape.warps_per_row == 8)
        return run.template operator()<1, 8>();
    throw std::invalid_argument("mxfp8 GEMV: unregistered row schedule");
}

// Blocks that keep every SM of the 70-SM RTX 5070 Ti busy with two resident blocks.
inline constexpr int kMxfp8TargetBlocks = 140;

// The widest schedule that still fills the device with blocks: wide rows-per-block amortise the
// activation reads, a K split keeps small-N problems (N = 48, 96, 640) from leaving SMs idle.
inline Mxfp8GemvShape select_mxfp8_gemv_shape(int rows) {
    constexpr Mxfp8GemvShape kOrder[]{{2, 1}, {1, 1}, {1, 2}, {1, 4}, {1, 8}};
    for (const auto shape : kOrder) {
        const int rows_per_block = kMxfp8GemvWarps / shape.warps_per_row * shape.rows_per_warp;
        if (div_up(rows, rows_per_block) >= kMxfp8TargetBlocks) return shape;
    }
    return kOrder[4];
}

template <class Output>
void launch_mxfp8_a16_gemv_output(const Mxfp8LinearOperands& op, Output output,
                                  cudaStream_t stream) {
    const Mxfp8GemvShape shape = select_mxfp8_gemv_shape(op.rows);
    switch (op.tokens) {
    case 1:
        return launch_mxfp8_a16_gemv_t<1>(op, output, shape, stream);
    case 2:
        return launch_mxfp8_a16_gemv_t<2>(op, output, shape, stream);
    case 3:
        return launch_mxfp8_a16_gemv_t<3>(op, output, shape, stream);
    case 4:
        return launch_mxfp8_a16_gemv_t<4>(op, output, shape, stream);
    case 5:
        return launch_mxfp8_a16_gemv_t<5>(op, output, shape, stream);
    case 6:
        return launch_mxfp8_a16_gemv_t<6>(op, output, shape, stream);
    case 7:
        return launch_mxfp8_a16_gemv_t<7>(op, output, shape, stream);
    case 8:
        return launch_mxfp8_a16_gemv_t<8>(op, output, shape, stream);
    default:
        throw std::invalid_argument("mxfp8 GEMV: T must be in [1,8]");
    }
}

} // namespace ninfer::ops::detail
