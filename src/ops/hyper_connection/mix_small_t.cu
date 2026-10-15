// Implements the small-T route (T <= 8) of hyper_connection_mix / hyper_connection_combine_mix
// (include/ninfer/ops/hyper_connection.h) over Q8_G32_FP16 row-split Down/Up weights, and over
// NVFP4 Down/Up weights (option, see the header).
//
// Two launches, no cooperative grid barrier (it needed one co-resident CTA per SM and blocked any
// overlap with other streams):
//   Down: one cluster of S = 4 CTAs per 16-row tile of Down, one CTA per lane (stream). The CTA's
//     16-row lane segment of Down arrives by one tensor-map copy into shared memory (128-byte
//     swizzle), requested together with the lane's norm weights before the launch waits for the
//     preceding kernel (it is a programmatic dependent: the weights do not depend on it). The
//     lane is committed (pending combine) and normalized in registers, already in the MMA operand
//     layout, contracted on FP16 m16n8k16 MMA with the tokens as N, and the four lane partials of
//     the tile are summed in fixed order by the cluster's rank 0 through distributed shared
//     memory, which writes low = silu(p/S) (FP16) and the inject gates.
//   Up + mix: programmatic dependent of Down, 16 hidden columns (64 Up rows) per CTA. Its Up rows
//     (one tensor-map copy per stream) and mix inputs are requested before the dependency wait;
//     then low, the Up contraction, the gate mix and the committed residual.
// The decomposition (Down split by lane, MMA with the tokens as N, exact magic-number widening of
// Q8 codes to FP16) comes from the fused BF16/FP8 hyper-connection kernel of ninfer6000
// (src/ops/launcher/hyperconnection.cu, Copyright Lars Karlslund, Apache-2.0).
//
// Operand layouts: Q8, lane (g, q) holds 8 consecutive K of each 32-K chunk (= one Q8 group over
// the four quads; the group scale multiplies the chunk's FP32 MMA sum). NVFP4, lane (g, q) holds
// K 4q..4q+3 of each 16-K group (one MMA = one group: its E4M3 scale multiplies the MMA sum; the
// matrix divisor divides the reduced projection). Both decode exactly (E2M1 and int8 are FP16
// values); the staged activations (normalized lane, low rank) are bounded, so FP16 is safe.
#include "ops/hyper_connection/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kS          = kHcStreams;
constexpr int kH          = kHcHidden;
constexpr int kSH         = kHcWide;
constexpr int kR          = kHcRank;
constexpr int kThreads    = 256;
constexpr int kWarps      = kThreads / 32;
constexpr int kDownWarpK  = kH / kWarps;     // 320 K per warp inside one lane
constexpr int kDownChunks = kDownWarpK / 32; // 10 Q8 groups per warp
constexpr int kUpKPad     = 384;             // row-split K_pad of K = 320
constexpr int kUpGroups   = kUpKPad / 32;    // 12 scale words per Up row
static_assert(kH % (kWarps * 32) == 0 && kR % 64 == 0);

// Eight signed Q8 codes widened exactly to FP16 without the conversion pipe: with u = code + 128
// in [1, 255], the FP16 word 0x6400 | u is 1024 + u, and subtracting 1152 leaves the code.
__device__ __forceinline__ std::uint32_t half2_minus_1152(std::uint32_t word) {
    const __half2 value =
        __hsub2(*reinterpret_cast<const __half2*>(&word),
                __halves2half2(__ushort_as_half(0x6480U), __ushort_as_half(0x6480U)));
    return *reinterpret_cast<const std::uint32_t*>(&value);
}

__device__ __forceinline__ uint4 widen_codes(const uint2 codes) {
    const std::uint32_t x = codes.x ^ 0x80808080U, y = codes.y ^ 0x80808080U;
    return make_uint4(half2_minus_1152(__byte_perm(x, 0x64646464U, 0x5140)),
                      half2_minus_1152(__byte_perm(x, 0x64646464U, 0x7362)),
                      half2_minus_1152(__byte_perm(y, 0x64646464U, 0x5140)),
                      half2_minus_1152(__byte_perm(y, 0x64646464U, 0x7362)));
}

__device__ __forceinline__ std::uint32_t pack_f16(float low, float high) {
    const __half2 value = __floats2half2_rn(low, high);
    return *reinterpret_cast<const std::uint32_t*>(&value);
}

__device__ __forceinline__ void mma_f16(float (&c)[4], std::uint32_t a0, std::uint32_t a1,
                                        std::uint32_t a2, std::uint32_t a3, std::uint32_t b0,
                                        std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One 32-wide K chunk: lane (g, q) holds 8 consecutive weights of rows g and g + 8 and the same
// 8 activations of token g. Dot products are permutation invariant, so the lane order maps onto
// the fragment K order directly: weights 0-3 feed the first MMA and 4-7 the second.
__device__ __forceinline__ void mma_chunk(float (&c)[4], const uint4& top, const uint4& bottom,
                                          const uint4& activation) {
    mma_f16(c, top.x, bottom.x, top.y, bottom.y, activation.x, activation.y);
    mma_f16(c, top.z, bottom.z, top.w, bottom.w, activation.z, activation.w);
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0F / (1.0F + expf(-x)); }

constexpr int kTile         = 16;
constexpr int kDownScaleRow = kH / 32; // 80 Q8 scale words per lane row segment
constexpr int kCols         = 16;      // hidden columns per Up CTA
constexpr int kUpThreads    = 128;
constexpr int kUpRows       = kS * kCols; // 64
constexpr int kLowStride16  = kR + 8;     // FP16 elements per low row in shared memory

// Operand layout of one format. kPieces x kWords 32-bit words per lane hold its K values.
template <bool Fp4>
struct HcLayout {
    static constexpr int kPieces = 10; // 32-K chunks per warp (320 K)
    static constexpr int kWords  = 4;  // 8 BF16/FP16 values

    __device__ static int k(int quad, int piece) { return 8 * quad + 32 * piece; }

    static constexpr int kRowBytes   = kH; // lane row segment bytes in Down
    static constexpr int kUpRowBytes = kR; // touched Up row bytes
};

template <>
struct HcLayout<true> {
    static constexpr int kPieces = 20; // 16-K groups per warp
    static constexpr int kWords  = 2;  // 4 values

    __device__ static int k(int quad, int piece) { return 4 * quad + 16 * piece; }

    static constexpr int kRowBytes   = kH / 2;
    static constexpr int kUpRowBytes = kR / 2;
};

// Down codes arrive as one tensor-map box per CTA, [16 rows][lines of 128 B], with the 128-byte
// swizzle (16-byte unit u of line L stored at unit u ^ (L mod 8)): the MMA operand reads of rows
// g = 0..7 then fall on several bank groups instead of one.
constexpr int kDownCodeBytes = kTile * kH; // Q8 size; NVFP4 uses half

struct alignas(1024) DownShared {
    std::int8_t codes[kDownCodeBytes]; // after the contraction: tile sums [16][8] FP32
    __half scales[kTile][kDownScaleRow];

    union {
        __nv_bfloat16 norm[kH];            // the lane's norm weights (activation phase)
        float warp_sums[kWarps][kTile][8]; // after the contraction
    } stage;

    float squares[kWarps][8];
    unsigned long long weight_barrier;
};

struct alignas(128) UpShared {
    std::int8_t codes[kUpRows][kR]; // four [16][row bytes] boxes, one per stream
    __half scales[kUpRows][kUpGroups];
    __half low[8][kLowStride16];
    float logits[kUpRows][8];
    float inverse_rms[8 * kS];
    unsigned long long barrier;
};

static_assert(sizeof(DownShared) <= 48 * 1024 && sizeof(UpShared) <= 48 * 1024);
static_assert(kTile * 8 * 4 <= kDownCodeBytes && sizeof(float) * kWarps * kTile * 8 <= kH * 2);
static_assert((kLowStride16 * 2) % 16 == 0 && (kCols * kR) % 128 == 0 &&
              (kCols * kR / 2) % 128 == 0);
static_assert(kH % kCols == 0 && kUpRows == 16 * (kUpThreads / 32) &&
              kHcFusedMaxT * kCols <= kUpThreads);

template <bool Fp4>
__device__ __forceinline__ int down_code_offset(int row, int byte) {
    const int line = row * (HcLayout<Fp4>::kRowBytes / 128) + byte / 128;
    return line * 128 + ((((byte >> 4) & 7) ^ (line & 7)) << 4) + (byte & 15);
}

__device__ __forceinline__ unsigned shared_address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void barrier_init(unsigned barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(barrier) : "memory");
}

__device__ __forceinline__ void barrier_init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void barrier_expect(unsigned barrier, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier),
                 "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void barrier_wait(unsigned barrier) {
    asm volatile("{\n .reg .pred done;\n WAIT_%=:\n"
                 " mbarrier.try_wait.parity.shared::cta.b64 done, [%0], 0;\n"
                 " @!done bra WAIT_%=;\n}\n" ::"r"(barrier)
                 : "memory");
}

__device__ __forceinline__ void bulk_copy(void* destination, const void* source, unsigned bytes,
                                          unsigned barrier) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::
            "r"(shared_address(destination)),
        "l"(source), "r"(bytes), "r"(barrier)
        : "memory");
}

__device__ __forceinline__ void tensor_load_2d(void* destination, const CUtensorMap* map, int x,
                                               int y, unsigned barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];\n" ::"r"(shared_address(destination)),
                 "l"(map), "r"(x), "r"(y), "r"(barrier)
                 : "memory");
}

__device__ __forceinline__ void tensor_load_3d(void* destination, const CUtensorMap* map, int x,
                                               int y, int z, unsigned barrier) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(shared_address(destination)),
                 "l"(map), "r"(x), "r"(y), "r"(z), "r"(barrier)
                 : "memory");
}

__device__ __forceinline__ float bf16_low(std::uint32_t pair) {
    return __uint_as_float(pair << 16);
}

__device__ __forceinline__ float bf16_high(std::uint32_t pair) {
    return __uint_as_float(pair & 0xFFFF0000U);
}

__device__ __forceinline__ float committed_value(float residual, float inject, float y) {
    return __bfloat162float(__float2bfloat16_rn(__fmaf_rn(inject, y, residual)));
}

template <int Words>
__device__ __forceinline__ void load_words(std::uint32_t (&out)[Words],
                                           const __nv_bfloat16* source) {
    if constexpr (Words == 4) {
        const uint4 v = *reinterpret_cast<const uint4*>(source);
        out[0]        = v.x;
        out[1]        = v.y;
        out[2]        = v.z;
        out[3]        = v.w;
    } else {
        const uint2 v = *reinterpret_cast<const uint2*>(source);
        out[0]        = v.x;
        out[1]        = v.y;
    }
}

// Two E2M1 codes (one byte) widened exactly to an FP16 pair (low nibble -> low half).
__device__ __forceinline__ std::uint32_t widen_e2m1_byte(std::uint32_t word, int byte) {
    std::uint32_t out;
    if (byte == 0)
        asm("{\n .reg .b8 b;\n mov.b32 {b, _, _, _}, %1;\n cvt.rn.f16x2.e2m1x2 %0, b;\n}\n"
            : "=r"(out)
            : "r"(word));
    else
        asm("{\n .reg .b8 b;\n mov.b32 {_, b, _, _}, %1;\n cvt.rn.f16x2.e2m1x2 %0, b;\n}\n"
            : "=r"(out)
            : "r"(word));
    return out;
}

// Four E4M3 scale bytes of one scale tile row (4 consecutive 16-K groups).
__device__ __forceinline__ float e4m3_scale(std::uint32_t word, int index) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>((word >> (8 * index)) & 0xFFU), __NV_E4M3);
    return __half2float(__half(h));
}

// Byte offset of the E4M3 scale word of (row, 4-group tile) in the BlockScaleK16M128x4 plane.
__device__ __forceinline__ std::int64_t fp4_scale_word(int row, int tile, int k_tiles) {
    return static_cast<std::int64_t>((row / 128) * k_tiles + tile) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4;
}

template <bool Fp4, int Tokens>
__global__ void __cluster_dims__(kS, 1, 1) __launch_bounds__(kThreads, 2)
    hc_down_kernel(const __grid_constant__ HcFusedMaps maps, HcFusedMix p) {
    using L = HcLayout<Fp4>;
    __shared__ DownShared sh;
    const int tid                 = static_cast<int>(threadIdx.x);
    const int lane                = tid & 31;
    const int warp                = tid >> 5;
    const int group               = lane >> 2;
    const int quad                = lane & 3;
    const int stream              = static_cast<int>(blockIdx.x) % kS; // = rank in the cluster
    const int tile                = static_cast<int>(blockIdx.x) / kS;
    const int row0                = tile * kTile;
    const unsigned weight_barrier = shared_address(&sh.weight_barrier);

    // Constant inputs first (they do not depend on the previous launch, so they are requested
    // while it finishes): the lane's norm weights (16-byte asynchronous copies), the NVFP4 scale
    // words of the lane's rows, and one tensor-map copy of the lane segment (+ Q8 scales).
    for (int i = tid; i < kH * 2 / 16; i += kThreads)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                         shared_address(sh.stage.norm + 8 * i)),
                     "l"(p.norm + kH * stream + 8 * i)
                     : "memory");
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    const bool has_token      = group < Tokens;
    const bool combine        = p.previous_y != nullptr;
    const int k_warp          = warp * kDownWarpK;
    constexpr int kScaleTiles = Fp4 ? kDownWarpK / 64 : 1; // 5 tiles of 4 groups per warp
    std::uint32_t scale_top[kScaleTiles], scale_bottom[kScaleTiles];
    if constexpr (Fp4) {
        const int first = (kH * stream + k_warp) / 64;
#pragma unroll
        for (int t = 0; t < kScaleTiles; ++t) {
            scale_top[t] = *reinterpret_cast<const std::uint32_t*>(
                p.down_fp4_scales + fp4_scale_word(row0 + group, first + t, kSH / 64));
            scale_bottom[t] = *reinterpret_cast<const std::uint32_t*>(
                p.down_fp4_scales + fp4_scale_word(row0 + group + 8, first + t, kSH / 64));
        }
    }
    if (tid == 0) {
        if ((shared_address(sh.codes) & 1023U) != 0) __trap(); // the swizzle needs 1 KiB alignment
        barrier_init(weight_barrier);
        barrier_init_fence();
        // Whole boxes count, rows past the matrix included (zero filled).
        constexpr unsigned kBytes = kTile * L::kRowBytes + (Fp4 ? 0 : kTile * kDownScaleRow * 2);
        barrier_expect(weight_barrier, kBytes);
        tensor_load_3d(sh.codes, &maps.down_codes, 0, (L::kRowBytes / 128) * stream, row0,
                       weight_barrier);
        if constexpr (!Fp4)
            tensor_load_2d(sh.scales, &maps.down_scales, kDownScaleRow * stream, row0,
                           weight_barrier);
    }
    // The residual, the pending block output and its gates come from earlier launches. The Up
    // launch is released only after this wait: it reads the same inputs before its own wait.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion(); // the Up launch reads our outputs only after its
                                               // wait
    std::uint32_t raw[L::kPieces][L::kWords], yv[L::kPieces][L::kWords];
    float inject = 0.0F;
    if (has_token) {
        const __nv_bfloat16* residual =
            p.residual + static_cast<std::int64_t>(kSH) * group + kH * stream + k_warp;
#pragma unroll
        for (int i = 0; i < L::kPieces; ++i) load_words(raw[i], residual + L::k(quad, i));
        if (combine) {
            inject                 = p.previous_inject[stream + kS * group];
            const __nv_bfloat16* y = p.previous_y + static_cast<std::int64_t>(kH) * group + k_warp;
#pragma unroll
            for (int i = 0; i < L::kPieces; ++i) load_words(yv[i], y + L::k(quad, i));
        }
    }

    std::uint32_t lane_x[L::kPieces][L::kWords];
    float square = 0.0F;
    if (has_token) {
#pragma unroll
        for (int i = 0; i < L::kPieces; ++i) {
#pragma unroll
            for (int w = 0; w < L::kWords; ++w) {
                if (combine) {
                    const __nv_bfloat162 v = __floats2bfloat162_rn(
                        __fmaf_rn(inject, bf16_low(yv[i][w]), bf16_low(raw[i][w])),
                        __fmaf_rn(inject, bf16_high(yv[i][w]), bf16_high(raw[i][w])));
                    lane_x[i][w] = *reinterpret_cast<const std::uint32_t*>(&v);
                } else {
                    lane_x[i][w] = raw[i][w];
                }
                const float a = bf16_low(lane_x[i][w]), b = bf16_high(lane_x[i][w]);
                square = fmaf(a, a, square);
                square = fmaf(b, b, square);
            }
        }
    }
    square += __shfl_xor_sync(0xffffffffU, square, 1);
    square += __shfl_xor_sync(0xffffffffU, square, 2);
    if (quad == 0 && has_token) sh.squares[warp][group] = square;
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads(); // square partials, norm weights, barrier initialized

    std::uint32_t activation[L::kPieces][L::kWords];
    if (has_token) {
        float sum = 0.0F;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) sum += sh.squares[w][group];
        const float inv = rsqrtf(sum / static_cast<float>(kH) + p.eps);
        if (tile == 0 && warp == 0 && quad == 0) p.square_sums[stream + kS * group] = sum;
#pragma unroll
        for (int i = 0; i < L::kPieces; ++i) {
            std::uint32_t n[L::kWords];
            if constexpr (L::kWords == 4) {
                const uint4 v =
                    *reinterpret_cast<const uint4*>(&sh.stage.norm[k_warp + L::k(quad, i)]);
                n[0] = v.x;
                n[1] = v.y;
                n[2] = v.z;
                n[3] = v.w;
            } else {
                const uint2 v =
                    *reinterpret_cast<const uint2*>(&sh.stage.norm[k_warp + L::k(quad, i)]);
                n[0] = v.x;
                n[1] = v.y;
            }
#pragma unroll
            for (int w = 0; w < L::kWords; ++w)
                activation[i][w] =
                    pack_f16(bf16_low(lane_x[i][w]) * inv * (1.0F + bf16_low(n[w])),
                             bf16_high(lane_x[i][w]) * inv * (1.0F + bf16_high(n[w])));
        }
    } else {
#pragma unroll
        for (int i = 0; i < L::kPieces; ++i)
#pragma unroll
            for (int w = 0; w < L::kWords; ++w) activation[i][w] = 0;
    }

    barrier_wait(weight_barrier);
    float acc[4] = {};
#pragma unroll
    for (int i = 0; i < L::kPieces; ++i) {
        float chunk[4] = {};
        if constexpr (Fp4) {
            const int byte          = (k_warp + L::k(quad, i)) / 2;
            const std::uint32_t top = *reinterpret_cast<const std::uint16_t*>(
                sh.codes + down_code_offset<true>(group, byte));
            const std::uint32_t bot = *reinterpret_cast<const std::uint16_t*>(
                sh.codes + down_code_offset<true>(group + 8, byte));
            mma_f16(chunk, widen_e2m1_byte(top, 0), widen_e2m1_byte(bot, 0),
                    widen_e2m1_byte(top, 1), widen_e2m1_byte(bot, 1), activation[i][0],
                    activation[i][1]);
            const float st = e4m3_scale(scale_top[i / 4], i % 4);
            const float sb = e4m3_scale(scale_bottom[i / 4], i % 4);
            acc[0]         = fmaf(chunk[0], st, acc[0]);
            acc[1]         = fmaf(chunk[1], st, acc[1]);
            acc[2]         = fmaf(chunk[2], sb, acc[2]);
            acc[3]         = fmaf(chunk[3], sb, acc[3]);
        } else {
            const int k = k_warp + L::k(quad, i);
            const uint2 top =
                *reinterpret_cast<const uint2*>(sh.codes + down_code_offset<false>(group, k));
            const uint2 bot =
                *reinterpret_cast<const uint2*>(sh.codes + down_code_offset<false>(group + 8, k));
            mma_chunk(
                chunk, widen_codes(top), widen_codes(bot),
                make_uint4(activation[i][0], activation[i][1], activation[i][2], activation[i][3]));
            const float st = __half2float(sh.scales[group][warp * kDownChunks + i]);
            const float sb = __half2float(sh.scales[group + 8][warp * kDownChunks + i]);
            acc[0]         = fmaf(chunk[0], st, acc[0]);
            acc[1]         = fmaf(chunk[1], st, acc[1]);
            acc[2]         = fmaf(chunk[2], sb, acc[2]);
            acc[3]         = fmaf(chunk[3], sb, acc[3]);
        }
    }
    __syncthreads(); // norm and codes are dead: their space holds the sums
    float (*warp_sums)[kTile][8]             = sh.stage.warp_sums;
    float (*tile_sums)[8]                    = reinterpret_cast<float (*)[8]>(sh.codes);
    warp_sums[warp][group][2 * quad]         = acc[0];
    warp_sums[warp][group][2 * quad + 1]     = acc[1];
    warp_sums[warp][group + 8][2 * quad]     = acc[2];
    warp_sums[warp][group + 8][2 * quad + 1] = acc[3];
    __syncthreads();
    if (tid < kTile * 8) {
        float sum = 0.0F;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) sum += warp_sums[w][tid / 8][tid % 8];
        tile_sums[tid / 8][tid % 8] = sum;
    }
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    cluster.sync();
    if (stream == 0 && tid < kTile * 8) {
        const int row   = row0 + tid / 8;
        const int token = tid % 8;
        if (token < Tokens && row < p.rows) {
            float lane_sum[kS];
#pragma unroll
            for (int s = 0; s < kS; ++s)
                lane_sum[s] = *cluster.map_shared_rank(&tile_sums[tid / 8][token], s);
            float v = 0.0F;
#pragma unroll
            for (int s = 0; s < kS; ++s) v += lane_sum[s];
            v *= 1.0F / kS;
            if constexpr (Fp4) v /= p.down_divisor;
            if (row < kR)
                p.low[token * kR + row] = __float2half_rn(v * sigmoidf_(v));
            else
                p.inject[(row - kR) + kS * token] = 2.0F * sigmoidf_(v);
        }
    }
    // Peers keep their shared memory until rank 0 has read it (the reads have returned: relaxed).
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;\n" ::
                     : "memory");
}

template <bool Fp4, int Tokens>
__global__ void __launch_bounds__(kUpThreads)
    hc_up_mix_kernel(const __grid_constant__ HcFusedMaps maps, HcFusedMix p) {
    using L = HcLayout<Fp4>;
    __shared__ UpShared sh;
    const int tid          = static_cast<int>(threadIdx.x);
    const int lane         = tid & 31;
    const int warp         = tid >> 5;
    const int group        = lane >> 2;
    const int quad         = lane & 3;
    const int column_begin = static_cast<int>(blockIdx.x) * kCols;
    const unsigned barrier = shared_address(&sh.barrier);

    cudaTriggerProgrammaticLaunchCompletion(); // dependents still wait for our completion
    // Up rows first: the weight stream is the long pole. Box s = stream s, columns
    // [column_begin, +16) -> rows 16s..16s+15 of the CTA; warp w contracts box w.
    if (tid == 0) {
        barrier_init(barrier);
        barrier_init_fence();
        barrier_expect(barrier, kUpRows * L::kUpRowBytes + (Fp4 ? 0 : kS * kCols * kUpGroups * 2));
#pragma unroll
        for (int s = 0; s < kS; ++s) {
            if constexpr (Fp4) {
                tensor_load_2d(sh.codes[0] + s * kCols * L::kUpRowBytes, &maps.up_codes, 0,
                               s * kH + column_begin, barrier);
            } else {
                tensor_load_3d(sh.codes[s * kCols], &maps.up_codes, 0, 0, s * kH + column_begin,
                               barrier);
                bulk_copy(sh.scales[s * kCols],
                          p.up_scales +
                              (static_cast<std::int64_t>(s) * kH + column_begin) * kUpGroups,
                          kCols * kUpGroups * 2, barrier);
            }
        }
    }
    const int top_row = warp * 16 + group; // local row; global row warp * kH + column_begin + group
    constexpr int kScaleTiles = Fp4 ? kR / 64 : 1;
    std::uint32_t scale_top[kScaleTiles], scale_bottom[kScaleTiles];
    if constexpr (Fp4) {
        const int global_top = warp * kH + column_begin + group;
#pragma unroll
        for (int t = 0; t < kScaleTiles; ++t) {
            scale_top[t] = *reinterpret_cast<const std::uint32_t*>(
                p.up_fp4_scales + fp4_scale_word(global_top, t, kR / 64));
            scale_bottom[t] = *reinterpret_cast<const std::uint32_t*>(
                p.up_fp4_scales + fp4_scale_word(global_top + 8, t, kR / 64));
        }
    }
    // Mix inputs (outputs of earlier launches, not of the Down launch).
    const bool has_item = tid < Tokens * kCols;
    const int token     = tid / kCols;
    const int column    = tid % kCols;
    const int d         = column_begin + column;
    float committed[kS], mix_norm[kS];
    if (has_item) {
        const float y =
            p.previous_y != nullptr
                ? __bfloat162float(p.previous_y[d + static_cast<std::int64_t>(kH) * token])
                : 0.0F;
#pragma unroll
        for (int s = 0; s < kS; ++s) {
            const float h =
                __bfloat162float(p.residual[static_cast<std::int64_t>(kSH) * token + kH * s + d]);
            committed[s] = p.previous_y != nullptr
                               ? committed_value(h, p.previous_inject[s + kS * token], y)
                               : h;
            mix_norm[s]  = 1.0F + __bfloat162float(p.norm[kH * s + d]);
        }
    }

    cudaGridDependencySynchronize(); // the Down launch has completed: low, square sums, inject
    for (int i = tid; i < Tokens * (kR / 8); i += kUpThreads) {
        const int t = i / (kR / 8), part = i % (kR / 8);
        *reinterpret_cast<uint4*>(&sh.low[t][part * 8]) =
            *reinterpret_cast<const uint4*>(p.low + t * kR + part * 8);
    }
    if (tid < Tokens * kS)
        sh.inverse_rms[tid] = rsqrtf(p.square_sums[tid] / static_cast<float>(kH) + p.eps);
    __syncthreads(); // also orders the barrier initialization before other threads wait on it
    barrier_wait(barrier);

    float acc[4] = {};
#pragma unroll
    for (int i = 0; i < L::kPieces; ++i) {
        const int k    = L::k(quad, i);
        float chunk[4] = {};
        if constexpr (Fp4) {
            const uint2 activation   = group < Tokens
                                           ? *reinterpret_cast<const uint2*>(&sh.low[group][k])
                                           : make_uint2(0, 0);
            const std::int8_t* codes = sh.codes[0];
            const std::uint32_t top =
                *reinterpret_cast<const std::uint16_t*>(codes + top_row * L::kUpRowBytes + k / 2);
            const std::uint32_t bot = *reinterpret_cast<const std::uint16_t*>(
                codes + (top_row + 8) * L::kUpRowBytes + k / 2);
            mma_f16(chunk, widen_e2m1_byte(top, 0), widen_e2m1_byte(bot, 0),
                    widen_e2m1_byte(top, 1), widen_e2m1_byte(bot, 1), activation.x, activation.y);
            const float st = e4m3_scale(scale_top[i / 4], i % 4);
            const float sb = e4m3_scale(scale_bottom[i / 4], i % 4);
            acc[0]         = fmaf(chunk[0], st, acc[0]);
            acc[1]         = fmaf(chunk[1], st, acc[1]);
            acc[2]         = fmaf(chunk[2], sb, acc[2]);
            acc[3]         = fmaf(chunk[3], sb, acc[3]);
        } else {
            const uint4 activation = group < Tokens
                                         ? *reinterpret_cast<const uint4*>(&sh.low[group][k])
                                         : make_uint4(0, 0, 0, 0);
            mma_chunk(chunk, widen_codes(*reinterpret_cast<const uint2*>(&sh.codes[top_row][k])),
                      widen_codes(*reinterpret_cast<const uint2*>(&sh.codes[top_row + 8][k])),
                      activation);
            const float st = __half2float(sh.scales[top_row][i]);
            const float sb = __half2float(sh.scales[top_row + 8][i]);
            acc[0]         = fmaf(chunk[0], st, acc[0]);
            acc[1]         = fmaf(chunk[1], st, acc[1]);
            acc[2]         = fmaf(chunk[2], sb, acc[2]);
            acc[3]         = fmaf(chunk[3], sb, acc[3]);
        }
    }
    if constexpr (Fp4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] /= p.up_divisor;
    }
    sh.logits[top_row][2 * quad]         = acc[0];
    sh.logits[top_row][2 * quad + 1]     = acc[1];
    sh.logits[top_row + 8][2 * quad]     = acc[2];
    sh.logits[top_row + 8][2 * quad + 1] = acc[3];
    __syncthreads();

    if (!has_item) return;
    float mixed = 0.0F;
#pragma unroll
    for (int s = 0; s < kS; ++s) {
        if (p.previous_y != nullptr)
            p.residual[static_cast<std::int64_t>(kSH) * token + kH * s + d] =
                __float2bfloat16_rn(committed[s]);
        const float normalized = committed[s] * sh.inverse_rms[token * kS + s] * mix_norm[s];
        mixed = fmaf(normalized, sigmoidf_(sh.logits[s * kCols + column][token]), mixed);
    }
    p.mixed[d + static_cast<std::int64_t>(kH) * token] = __float2bfloat16_rn(mixed * (1.0F / kS));
}

using SplitKernel = void (*)(HcFusedMaps, HcFusedMix);

template <bool Fp4, int... Tokens>
constexpr auto down_kernels(std::integer_sequence<int, Tokens...>) {
    return std::array<SplitKernel, sizeof...(Tokens)>{hc_down_kernel<Fp4, Tokens + 1>...};
}

template <bool Fp4, int... Tokens>
constexpr auto up_kernels(std::integer_sequence<int, Tokens...>) {
    return std::array<SplitKernel, sizeof...(Tokens)>{hc_up_mix_kernel<Fp4, Tokens + 1>...};
}

constexpr auto kDownKernels = down_kernels<false>(std::make_integer_sequence<int, kHcFusedMaxT>{});
constexpr auto kUpKernels   = up_kernels<false>(std::make_integer_sequence<int, kHcFusedMaxT>{});
constexpr auto kDownKernelsF4 = down_kernels<true>(std::make_integer_sequence<int, kHcFusedMaxT>{});
constexpr auto kUpKernelsF4   = up_kernels<true>(std::make_integer_sequence<int, kHcFusedMaxT>{});

CUtensorMap encode_map(CUtensorMapDataType type, const void* base, int rank,
                       const std::uint64_t* dims, const std::uint64_t* strides,
                       const std::uint32_t* box, CUtensorMapSwizzle swizzle) {
    CUtensorMap map{};
    const std::uint32_t steps[3]{1, 1, 1};
    const CUresult status =
        cuTensorMapEncodeTiled(&map, type, static_cast<cuuint32_t>(rank), const_cast<void*>(base),
                               dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("hyper_connection TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return map;
}

HcFusedMaps make_maps(const HcFusedMix& p) {
    HcFusedMaps maps{};
    if (p.fp4) {
        { // Down codes [384][40 lines][128 B]; box = one lane segment (10 lines) of a 16-row tile.
            const std::uint64_t dims[3]{128, kSH / 2 / 128,
                                        static_cast<std::uint64_t>(p.down_padded_rows)};
            const std::uint64_t strides[2]{128, kSH / 2};
            const std::uint32_t box[3]{128, kH / 2 / 128, kTile};
            maps.down_codes = encode_map(CU_TENSOR_MAP_DATA_TYPE_UINT8, p.down_codes, 3, dims,
                                         strides, box, CU_TENSOR_MAP_SWIZZLE_128B);
        }
        { // Up codes [10240][160 B]; box = 16 rows.
            const std::uint64_t dims[2]{kR / 2, kSH};
            const std::uint64_t strides[1]{kR / 2};
            const std::uint32_t box[2]{kR / 2, kCols};
            maps.up_codes = encode_map(CU_TENSOR_MAP_DATA_TYPE_UINT8, p.up_codes, 2, dims, strides,
                                       box, CU_TENSOR_MAP_SWIZZLE_NONE);
        }
        return maps;
    }
    { // Down codes [rows][80 lines][128 B]; box = one lane segment (20 lines) of a 16-row tile.
        const std::uint64_t dims[3]{128, kSH / 128, static_cast<std::uint64_t>(p.rows)};
        const std::uint64_t strides[2]{128, kSH};
        const std::uint32_t box[3]{128, kH / 128, kTile};
        maps.down_codes = encode_map(CU_TENSOR_MAP_DATA_TYPE_UINT8, p.down_codes, 3, dims, strides,
                                     box, CU_TENSOR_MAP_SWIZZLE_128B);
    }
    { // Down scales [rows][320] FP16; box = 80 words of 16 rows.
        const std::uint64_t dims[2]{kSH / 32, static_cast<std::uint64_t>(p.rows)};
        const std::uint64_t strides[1]{kSH / 32 * 2};
        const std::uint32_t box[2]{kDownScaleRow, kTile};
        maps.down_scales = encode_map(CU_TENSOR_MAP_DATA_TYPE_UINT16, p.down_scales, 2, dims,
                                      strides, box, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
    { // Up codes [10240][6 x 64 B] (K padded to 384); box = the 320 touched bytes of 16 rows.
        const std::uint64_t dims[3]{64, kUpKPad / 64, kSH};
        const std::uint64_t strides[2]{64, kUpKPad};
        const std::uint32_t box[3]{64, kR / 64, kCols};
        maps.up_codes = encode_map(CU_TENSOR_MAP_DATA_TYPE_UINT8, p.up_codes, 3, dims, strides, box,
                                   CU_TENSOR_MAP_SWIZZLE_NONE);
    }
    return maps;
}

} // namespace

bool hyper_connection_fused_supported() {
    // Per device: thread block clusters (sm_90+) are a device fact, not request state.
    constexpr int kMaxDevices = 16;
    static int support[kMaxDevices]; // 0 unknown, 1 yes, 2 no
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) return false;
    if (support[device] == 0) {
        int cluster = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&cluster, cudaDevAttrClusterLaunch, device));
        support[device] = cluster != 0 ? 1 : 2;
    }
    return support[device] == 1;
}

std::size_t hyper_connection_fused_workspace_bytes(std::int32_t tokens) {
    const auto t = static_cast<std::size_t>(tokens);
    return (t * kS * 4 + 255) / 256 * 256 + t * kR * 2; // square sums FP32 [T,S], low FP16 [T,R]
}

void hyper_connection_fused_mix_launch(const HcFusedMix& params, cudaStream_t stream) {
    const int tiles        = (params.rows + kTile - 1) / kTile;
    const HcFusedMaps maps = make_maps(params);
    const auto& down       = params.fp4 ? kDownKernelsF4 : kDownKernels;
    const auto& up         = params.fp4 ? kUpKernelsF4 : kUpKernels;
    CUDA_CHECK(pdl::launch_dependent(pdl::LaunchConfig{dim3(tiles * kS), dim3(kThreads), 0, stream},
                                     down[params.tokens - 1], maps, params));
    CUDA_CHECK(
        pdl::launch_dependent(pdl::LaunchConfig{dim3(kH / kCols), dim3(kUpThreads), 0, stream},
                              up[params.tokens - 1], maps, params));
}

} // namespace ninfer::ops::detail
