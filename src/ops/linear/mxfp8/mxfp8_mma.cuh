#pragma once

// MXFP8 x BF16 Tensor Core GEMM (prefill, and T = 6..8 on the wide problems).
//
// out[N,T] = W[N,K] * x[K,T]. One K tile holds KGroups 32-wide scale groups. Raw E4M3 codes and
// BF16 activations are staged with a multi-stage cp.async pipeline, together with the 4-byte word
// of each row's UE8M0 scales that holds the tile's exponents. Each warp widens its A fragments
// straight from the code tile with integer operations (mxfp8_e4m3x2_to_bf16x2_scaled_bits: no
// conversion-pipe instructions), applies the exact power-of-two scale with one BF16 multiply, and
// issues m16n8k16 BF16 MMA with FP32 accumulation. The decoded weights are bit-identical to
// e4m3(code) * 2^(scale - 127) rounded once to BF16.
//
// Tiles are walked token-tile-fastest (a 1-D grid), so the CTAs that share one weight row block run
// together and the weight streams from DRAM about once.
//
// Requires K % (32 KGroups) == 0, K/32 % 4 == 0 and a 4-byte aligned scale plane (true for every
// registered problem): each stage copies the aligned scale word that holds its groups.
//
// K order inside each 16-wide MMA step is permuted identically for A and B (the contraction is
// order-free): fragment slots {2q, 2q+1, 2q+8, 2q+9} of thread q = lane % 4 hold the physical
// columns {4q, 4q+1, 4q+2, 4q+3}, so A is one 32-bit and B one 64-bit shared load per fragment.

#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/launcher/kernel_attr_once.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/mxfp8/mxfp8_codec.cuh"
#include "ops/linear/mxfp8/mxfp8_operands.h"

#include <stdexcept>

namespace ninfer::ops::detail {

template <int BM_, int BN_, int WarpsM_, int WarpsN_, int Stages_, int KGroups_ = 1,
          int MinBlocks_ = 1, bool TokenFastest_ = true>
struct Mxfp8MmaConfig {
    static constexpr bool TokenFastest = TokenFastest_;
    static constexpr int BM            = BM_;
    static constexpr int BN            = BN_;
    static constexpr int WarpsM        = WarpsM_;
    static constexpr int WarpsN        = WarpsN_;
    static constexpr int Stages        = Stages_;
    static constexpr int KGroups       = KGroups_;
    static constexpr int MinBlocks     = MinBlocks_;
    static constexpr int BK            = 32 * KGroups;
    static constexpr int Threads       = WarpsM * WarpsN * 32;
    static constexpr int WTM           = BM / WarpsM;
    static constexpr int WTN           = BN / WarpsN;
    static constexpr int MT            = WTM / 16;
    static constexpr int NT            = WTN / 8;
    // Per stage: BM x BK code bytes, BN x BK BF16 activations, and the aligned 4-byte word of
    // each row's scale plane that holds this K tile's exponents.
    static constexpr int AStageBytes = BM * BK;
    static constexpr int BStageBytes = BN * BK * 2;
    static constexpr int SStageBytes = BM * 4;
    static constexpr int StageBytes  = AStageBytes + BStageBytes + SStageBytes;
    static constexpr int SharedBytes = Stages * StageBytes;
    static_assert(BM % (16 * WarpsM) == 0 && BN % (8 * WarpsN) == 0 && Stages >= 2);
    static_assert(KGroups == 1 || KGroups == 2 || KGroups == 4);
    static_assert(SharedBytes <= 99 * 1024);
};

// Unpadded tiles with XOR-swizzled 16-byte chunks: the 32-bit A loads (8 rows x 16 bytes) and
// the 64-bit B loads (4 tokens x 32 bytes per half-warp) of one MMA step are bank-conflict free.
template <int KGroups>
__device__ __forceinline__ int mxfp8_a_offset(int row, int chunk) {
    constexpr int kRowsPerLine = 4 / KGroups; // code rows per 128-byte bank line
    return row * 32 * KGroups + ((chunk ^ (row / kRowsPerLine)) & (2 * KGroups - 1)) * 16;
}

template <int KGroups>
__device__ __forceinline__ int mxfp8_b_offset(int row, int chunk) {
    const int swizzle = KGroups == 1 ? (row & 2) : ((row & 3) << 1);
    return row * 64 * KGroups + (chunk ^ swizzle) * 16;
}

template <class Cfg, class Output>
__global__ __launch_bounds__(Cfg::Threads,
                             Cfg::MinBlocks) void mxfp8_a16_mma_kernel(Mxfp8LinearOperands op,
                                                                       Output output) {
    constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, MT = Cfg::MT, NT = Cfg::NT;
    constexpr int KG = Cfg::KGroups;
    extern __shared__ __align__(16) unsigned char shared[];

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = warp / Cfg::WarpsN;
    const int wn   = warp % Cfg::WarpsN;
    const int gid  = lane >> 2;
    const int quad = lane & 3;

    const int m = op.rows, k = op.k, tokens = op.tokens;
    const int groups  = k / 32;
    const int k_tiles = k / BK;
    const int tile    = static_cast<int>(blockIdx.x);
    const int n_tiles = div_up(tokens, BN);
    const int m_tiles = div_up(m, BM);
    const int m0      = (Cfg::TokenFastest ? tile / n_tiles : tile % m_tiles) * BM;
    const int n0      = (Cfg::TokenFastest ? tile % n_tiles : tile / m_tiles) * BN;

    auto stage_base = [&](int stage) { return shared + stage * Cfg::StageBytes; };

    auto load_stage = [&](int stage, int kt) {
        unsigned char* a  = stage_base(stage);
        unsigned char* b  = a + Cfg::AStageBytes;
        unsigned char* sc = b + Cfg::BStageBytes;
        for (int item = tid; item < BM * 2 * KG; item += Cfg::Threads) {
            const int row   = item / (2 * KG);
            const int chunk = item % (2 * KG);
            const bool live = m0 + row < m;
            const auto* src = op.codes + static_cast<std::int64_t>(live ? m0 + row : 0) * k +
                              kt * BK + chunk * 16;
            cp_async_zfill<16, Cache::cg>(a + mxfp8_a_offset<KG>(row, chunk), src, live ? 16 : 0);
        }
        for (int item = tid; item < BN * 4 * KG; item += Cfg::Threads) {
            const int token = item / (4 * KG);
            const int chunk = item % (4 * KG);
            const bool live = n0 + token < tokens;
            const auto* src =
                op.x + static_cast<std::int64_t>(live ? n0 + token : 0) * k + kt * BK + chunk * 8;
            cp_async_zfill<16>(b + mxfp8_b_offset<KG>(token, chunk), src, live ? 16 : 0);
        }
        for (int row = tid; row < BM; row += Cfg::Threads) {
            const bool live = m0 + row < m;
            const auto* src = op.scales + static_cast<std::int64_t>(live ? m0 + row : 0) * groups +
                              ((kt * KG) & ~3);
            cp_async_zfill<4>(sc + row * 4, src, live ? 4 : 0);
        }
    };

#pragma unroll
    for (int s = 0; s < Cfg::Stages - 1; ++s) {
        if (s < k_tiles) load_stage(s, s);
        cp_commit();
    }

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[mi][ni][e] = 0.0f;

    const int a_row_base = wm * Cfg::WTM + gid;
    const int b_tok_base = wn * Cfg::WTN + gid;

#pragma unroll 1
    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_wait<Cfg::Stages - 2>();
        __syncthreads();
        {
            const int next = kt + Cfg::Stages - 1;
            if (next < k_tiles) load_stage(next % Cfg::Stages, next);
            cp_commit();
        }
        const unsigned char* a  = stage_base(kt % Cfg::Stages);
        const unsigned char* b  = a + Cfg::AStageBytes;
        const unsigned char* sc = b + Cfg::BStageBytes;

#pragma unroll
        for (int g = 0; g < KG; ++g) {
            const int byte = (kt * KG + g) & 3;
            unsigned s_lo[MT], s_hi[MT];
            bool direct = true;
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = a_row_base + mi * 16;
                s_lo[mi]      = sc[row * 4 + byte];
                s_hi[mi]      = sc[(row + 8) * 4 + byte];
                direct = direct && mxfp8_ue8m0_direct(s_lo[mi]) && mxfp8_ue8m0_direct(s_hi[mi]);
            }

            // Warp-uniform: every warp of a real checkpoint takes the one-multiply decode; the
            // two-multiply copy only runs for tiles holding a weight >= 2^8 * 448.
            const auto mma_group = [&]<bool Direct>() {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int ks = g * 2 + h;
                    unsigned af[MT][4];
#pragma unroll
                    for (int mi = 0; mi < MT; ++mi) {
                        const int row     = a_row_base + mi * 16;
                        const unsigned lo = *reinterpret_cast<const unsigned*>(
                            a + mxfp8_a_offset<KG>(row, ks) + quad * 4);
                        const unsigned hi = *reinterpret_cast<const unsigned*>(
                            a + mxfp8_a_offset<KG>(row + 8, ks) + quad * 4);
                        const unsigned codes[4] = {mxfp8_e4m3x2_to_bf16x2_shifted_bits<0>(lo),
                                                   mxfp8_e4m3x2_to_bf16x2_shifted_bits<0>(hi),
                                                   mxfp8_e4m3x2_to_bf16x2_shifted_bits<1>(lo),
                                                   mxfp8_e4m3x2_to_bf16x2_shifted_bits<1>(hi)};
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const unsigned s = (e & 1) ? s_hi[mi] : s_lo[mi];
                            if constexpr (Direct)
                                af[mi][e] = mxfp8_bf16x2_mul_bits(
                                    codes[e], mxfp8_ue8m0_to_shifted_bf16x2_bits(s));
                            else
                                af[mi][e] = mxfp8_scale_shifted_bf16x2_bits(codes[e], s);
                        }
                    }
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        const int token = b_tok_base + ni * 8;
                        // K columns [16 ks + 4 quad, +4) are bytes [32 ks + 8 quad, +8) of the row.
                        const uint2 bf = *reinterpret_cast<const uint2*>(
                            b + mxfp8_b_offset<KG>(token, ks * 2 + (quad >> 1)) + (quad & 1) * 8);
#pragma unroll
                        for (int mi = 0; mi < MT; ++mi)
                            mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                                     af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf.x, bf.y);
                    }
                }
            };
            if (__all_sync(0xffffffffu, direct))
                mma_group.template operator()<true>();
            else
                mma_group.template operator()<false>();
        }
    }
    cp_wait<0>();

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int row = m0 + a_row_base + mi * 16;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int token = n0 + wn * Cfg::WTN + ni * 8 + quad * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = row + (e >> 1) * 8;
                const int t = token + (e & 1);
                if (r < m && t < tokens) output.store(r, t, acc[mi][ni][e]);
            }
        }
    }
}

template <class Cfg, class Output>
void launch_mxfp8_a16_mma_cfg(const Mxfp8LinearOperands& op, Output output, cudaStream_t stream) {
    constexpr auto kernel = mxfp8_a16_mma_kernel<Cfg, Output>;
    if constexpr (Cfg::SharedBytes > 48 * 1024) {
        // Per device: an attribute set only on the first device leaves the other rank at 48 KiB.
        static FuncAttrPerDevice attribute;
        attribute.ensure(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::SharedBytes);
    }
    const int blocks = div_up(op.rows, Cfg::BM) * div_up(op.tokens, Cfg::BN);
    kernel<<<blocks, Cfg::Threads, Cfg::SharedBytes, stream>>>(op, output);
    CUDA_CHECK(cudaGetLastError());
}

// Measured on the RTX 5070 Ti over the 11 registered shapes (cold L2, T = 6..512):
// - Big (128 x 128, 8 warps of 32 x 64, two scale groups per stage, 2 CTAs/SM) once it gives two
//   CTAs per SM: N >= 6656 at T = 512 (~89 TFLOPS at N = 13312); Big96 (96 x 128, 8 warps of
//   48 x 32) where Big's last wave would be 60-90 % full.
// - Tall (256 x 64) / Mid (128 x 64) for N >= 6656 at T <= 256, where 128-token tiles would be
//   half empty or leave SMs idle.
// - Square (64 x 64) for N <= 2560 at T >= 256, Thin (32 x 64) and Small (16 x 32) for the narrow
//   problems at smaller T, which need many CTAs to keep the K loop latency hidden.
// - Tiny (16 x 8, one warp, four scale groups per stage) for T = 6..8 on N >= 2560 rows, where it
//   streams the weight faster than the FP32 GEMV (see mxfp8_uses_gemv).
using Mxfp8MmaBig    = Mxfp8MmaConfig<128, 128, 4, 2, 2, 2, 2>;
using Mxfp8MmaBig96  = Mxfp8MmaConfig<96, 128, 2, 4, 2, 2, 2>;
using Mxfp8MmaTall   = Mxfp8MmaConfig<256, 64, 8, 1, 2, 2, 2>;
using Mxfp8MmaMid    = Mxfp8MmaConfig<128, 64, 4, 1, 3, 2, 2>;
using Mxfp8MmaSquare = Mxfp8MmaConfig<64, 64, 2, 2, 4, 1, 4>;
using Mxfp8MmaThin   = Mxfp8MmaConfig<32, 64, 2, 2, 4, 2, 4>;
using Mxfp8MmaSmall  = Mxfp8MmaConfig<16, 32, 1, 4, 4, 2, 8, false>;
using Mxfp8MmaTiny   = Mxfp8MmaConfig<16, 8, 1, 1, 4, 4, 16, false>;

inline constexpr int kMxfp8Sms = 70;

template <class Output>
void launch_mxfp8_a16_mma_output(const Mxfp8LinearOperands& op, Output output,
                                 cudaStream_t stream) {
    const auto blocks = [&]<class Cfg>() {
        return div_up(op.rows, Cfg::BM) * div_up(op.tokens, Cfg::BN);
    };
    const int t = op.tokens;
    if (t <= Mxfp8MmaTiny::BN) return launch_mxfp8_a16_mma_cfg<Mxfp8MmaTiny>(op, output, stream);
    if (const int big = blocks.template operator()<Mxfp8MmaBig>(); big >= 2 * kMxfp8Sms) {
        // A last wave 60-90 % full (N = 8192, 16384 at T = 512) ends sooner with the 96-row tile.
        const int last = big % (2 * kMxfp8Sms);
        if (last * 10 > 12 * kMxfp8Sms && last * 10 < 18 * kMxfp8Sms)
            return launch_mxfp8_a16_mma_cfg<Mxfp8MmaBig96>(op, output, stream);
        return launch_mxfp8_a16_mma_cfg<Mxfp8MmaBig>(op, output, stream);
    }
    if (t <= 256 && op.rows >= 4096 && blocks.template operator()<Mxfp8MmaTall>() >= 48)
        return launch_mxfp8_a16_mma_cfg<Mxfp8MmaTall>(op, output, stream);
    if (t <= 256 && op.rows >= 4096 && blocks.template operator()<Mxfp8MmaMid>() >= 48)
        return launch_mxfp8_a16_mma_cfg<Mxfp8MmaMid>(op, output, stream);
    if (t >= 256 && blocks.template operator()<Mxfp8MmaSquare>() >= kMxfp8Sms)
        return launch_mxfp8_a16_mma_cfg<Mxfp8MmaSquare>(op, output, stream);
    if (blocks.template operator()<Mxfp8MmaThin>() >= 40)
        return launch_mxfp8_a16_mma_cfg<Mxfp8MmaThin>(op, output, stream);
    return launch_mxfp8_a16_mma_cfg<Mxfp8MmaSmall>(op, output, stream);
}

} // namespace ninfer::ops::detail
