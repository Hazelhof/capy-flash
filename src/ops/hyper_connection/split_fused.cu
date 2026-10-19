// Implements the fused route of hyper_connection_mix_split / hyper_connection_combine_mix_split
// (include/ninfer/ops/tp2/hyper_connection.h) for T <= 8 inside a two-device capture on the
// pinned-host mailbox: one cooperative launch per rank, the two cross-rank sums exchanged
// mid-kernel through PeerMailbox::exchange_slot with the pipelined protocol (peer_exchange_lane).
//
// The structure is that of T10's cooperative Q8 mix (the former mix_q8.cu, replaced for the
// replicated mix by the two-launch route of mix_small_t.cu; ninfer6000's fused hyper-connection
// design, Lars Karlslund, Apache-2.0) over the rank's two streams. This kernel is self-contained
// (it shares only launch.h's geometry constants) and keeps the cooperative launch: its two
// cross-rank exchanges sit between grid barriers of the one launch:
//   phase 1, CTA (local stream, slice): pending combine of the lane, its RMS and normalized FP16
//            activations, lane-segment partial dots of up to two 8-row Down tiles;
//   grid barrier; exchange 1: warps [0, lanes1) sum the two local streams' partials of the
//            projection, exchange them with the peer (FP32) and store p = fl32(p_0 + p_1);
//   grid barrier; phase 2: silu(p/S) in FP16, the rank's inject gates (CTA 0), Up rows of the CTA's
//            hidden columns for the two local streams, the FP32 collapse partial of its columns
//            (and the committed residual);
//   grid barrier; exchange 2: warps [0, lanes2) exchange the collapse partial and store
//            mixed = RNE(fl32(q_0 + q_1) / S).
// Both ranks run the same exchange sequence on the same slots at the pipelined geometry, so the
// mailbox's own exchanges interleave with these (SLOT REUSE holds: an exchange's publication on a
// rank follows its previous exchange's reads through the grid barriers).
#include "ops/hyper_connection/split_launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/peer_exchange_lane.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <array>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kS           = kHcStreams;     // streams of the model
constexpr int kLS          = kHcStreams / 2; // streams of a rank
constexpr int kH           = kHcHidden;
constexpr int kLW          = kLS * kH; // the rank's residual rows
constexpr int kR           = kHcRank;
constexpr int kThreads     = 256;
constexpr int kWarps       = kThreads / 32;
constexpr int kPerThread   = kH / kThreads;   // 10
constexpr int kDownWarpK   = kH / kWarps;     // 320 K per warp inside one lane
constexpr int kDownChunks  = kDownWarpK / 32; // 10 Q8 groups per warp
constexpr int kDownGroups  = kLW / 32;        // scale words per Down row of the shard
constexpr int kUpKPad      = 384;
constexpr int kUpGroups    = kUpKPad / 32;
constexpr int kUpChunks    = kR / 2 / 32;
constexpr int kActStride   = kH + 32;
constexpr int kLowStride   = kR + 32;
constexpr int kMinGrid     = 56;
constexpr int kMaxTiles    = 2;
constexpr int kUpPasses    = 2;
constexpr int kMaxColumns  = (kH + kMinGrid - 1) / kMinGrid; // 46
constexpr int kMaxUpRows   = kLS * kMaxColumns;              // 92
constexpr int kVecsPerLane = kPeerPipelinedVecsPerLane;      // 2: one lane = 1 KiB
static_assert((kHcRowsCap + 7) / 8 <= kMaxTiles * (kMinGrid / kLS));
static_assert(kMaxUpRows <= kUpPasses * 64);
static_assert(kHcFusedMaxT * kMaxColumns <= 2 * kThreads);
static_assert(kHcFusedMaxT * kActStride * 2 <= 48 * 1024);
static_assert(8 * kLowStride * 2 + 2 * kMaxUpRows * 8 * 4 <= 48 * 1024);

__device__ __forceinline__ uint2 load_streaming_u2(const std::int8_t* pointer) {
    uint2 bits;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(bits.x), "=r"(bits.y)
                 : "l"(pointer));
    return bits;
}

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

__device__ __forceinline__ void mma_chunk(float (&c)[4], const uint4& top, const uint4& bottom,
                                          const uint4& activation) {
    mma_f16(c, top.x, bottom.x, top.y, bottom.y, activation.x, activation.y);
    mma_f16(c, top.z, bottom.z, top.w, bottom.w, activation.z, activation.w);
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0F / (1.0F + expf(-x)); }

// One exchange of `vecs` 16-byte FP32 vectors of `source` (this rank's partial) with the peer;
// warps [0, lanes) of the grid take part. `store(v, sum)` receives each summed vector.
template <class Store>
__device__ __forceinline__ void grid_exchange(const PeerVec* source, const HcSplitExchange& slot,
                                              std::uint32_t* hang, int vecs, Store store) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp * (32 * kVecsPerLane) >= vecs) { return; } // warp-uniform
    const int base             = warp * (32 * kVecsPerLane) + lane;
    PeerVec mine[kVecsPerLane] = {};
#pragma unroll
    for (int j = 0; j < kVecsPerLane; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { mine[j] = source[v]; }
    }
    PeerVec peer[kVecsPerLane] = {};
    if (!peer_exchange_lane<kVecsPerLane>(
            warp, lane, mine, peer, static_cast<PeerVec*>(slot.mine_payload), slot.mine_flags,
            static_cast<const PeerVec*>(slot.peer_payload), slot.peer_flags, slot.epochs, hang,
            vecs, kPeerSpinLimit, kPeerPollSleepNs)) {
        return;
    }
#pragma unroll
    for (int j = 0; j < kVecsPerLane; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { store(v, peer_sum_f32(mine[j], peer[j]).value); }
    }
}

template <int Tokens>
__global__ void __launch_bounds__(kThreads, 1) hc_split_fused_kernel(HcSplitFused p) {
    extern __shared__ float4 split_shared_storage[];
    auto* activations = reinterpret_cast<__half*>(split_shared_storage); // [Tokens,kActStride]
    __shared__ float square_shared[kWarps][Tokens];
    __shared__ float tile_shared[kWarps][8][8];
    __shared__ float inverse_rms[Tokens * kLS];

    const int tid   = static_cast<int>(threadIdx.x);
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int group = lane >> 2;
    const int quad  = lane & 3;

    const int slices = static_cast<int>(gridDim.x) / kLS;
    const int stream = static_cast<int>(blockIdx.x) % kLS; // local stream
    const int slice  = static_cast<int>(blockIdx.x) / kLS;
    const int rows   = p.rows;
    const int tiles  = (rows + 7) / 8;

    float x[Tokens][kPerThread];
    const std::int64_t lane_base = static_cast<std::int64_t>(kH) * stream + tid;
    {
        __nv_bfloat16 h[Tokens][kPerThread];
#pragma unroll
        for (int token = 0; token < Tokens; ++token)
#pragma unroll
            for (int j = 0; j < kPerThread; ++j)
                h[token][j] =
                    p.residual[lane_base + static_cast<std::int64_t>(kLW) * token + j * kThreads];
        if (p.previous_y != nullptr) {
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                const float scale = p.previous_inject[stream + kLS * token];
#pragma unroll
                for (int j = 0; j < kPerThread; ++j)
                    x[token][j] = __bfloat162float(__float2bfloat16_rn(
                        __bfloat162float(h[token][j]) +
                        scale * __bfloat162float(p.previous_y[tid + j * kThreads + kH * token])));
            }
        } else {
#pragma unroll
            for (int token = 0; token < Tokens; ++token)
#pragma unroll
                for (int j = 0; j < kPerThread; ++j) x[token][j] = __bfloat162float(h[token][j]);
        }
    }

    uint2 bits[kMaxTiles][kDownChunks];
    __half scales[kMaxTiles][kDownChunks];
#pragma unroll
    for (int i = 0; i < kMaxTiles; ++i) {
        const int row = (slice + i * slices) * 8 + group;
        if (row >= rows) {
#pragma unroll
            for (int c = 0; c < kDownChunks; ++c) {
                bits[i][c]   = make_uint2(0, 0);
                scales[i][c] = __float2half(0.0F);
            }
            continue;
        }
        const std::int8_t* codes = p.down_codes + static_cast<std::int64_t>(row) * kLW +
                                   kH * stream + warp * kDownWarpK + 8 * quad;
        const __half* scale = p.down_scales + static_cast<std::int64_t>(row) * kDownGroups +
                              (kH / 32) * stream + warp * kDownChunks;
#pragma unroll
        for (int c = 0; c < kDownChunks; ++c) {
            bits[i][c]   = load_streaming_u2(codes + 32 * c);
            scales[i][c] = scale[c];
        }
    }

    {
        float square[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            square[token] = 0.0F;
#pragma unroll
            for (int j = 0; j < kPerThread; ++j)
                square[token] = fmaf(x[token][j], x[token][j], square[token]);
        }
#pragma unroll
        for (int shift = 16; shift; shift >>= 1)
#pragma unroll
            for (int token = 0; token < Tokens; ++token)
                square[token] += __shfl_xor_sync(0xffffffffU, square[token], shift);
        if (lane == 0) {
#pragma unroll
            for (int token = 0; token < Tokens; ++token) square_shared[warp][token] = square[token];
        }
        __syncthreads();
        float inv[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            float sum = 0.0F;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) sum += square_shared[w][token];
            inv[token] = rsqrtf(sum / static_cast<float>(kH) + p.eps);
            if (slice == 0 && tid == 0) p.square_sums[stream + kLS * token] = sum;
        }
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            const float norm_scale =
                1.0F + __bfloat162float(p.norm[kH * stream + tid + j * kThreads]);
#pragma unroll
            for (int token = 0; token < Tokens; ++token)
                activations[token * kActStride + tid + j * kThreads] =
                    __float2half_rn(x[token][j] * inv[token] * norm_scale);
        }
    }
    __syncthreads();

    const int columns_per_cta =
        (kH + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int column_begin = static_cast<int>(blockIdx.x) * columns_per_cta;
    const int columns      = max(0, min(columns_per_cta, kH - column_begin));
    const int up_rows      = kLS * columns;
    const int up_tile      = warp >> 1;
    const int up_half      = warp & 1;
    uint2 up_top[kUpPasses][kUpChunks], up_bottom[kUpPasses][kUpChunks];
    __half up_scale_top[kUpPasses][kUpChunks], up_scale_bottom[kUpPasses][kUpChunks];
    const auto load_up = [&](int local_row, uint2(&out)[kUpChunks], __half(&scale)[kUpChunks]) {
        if (local_row >= up_rows) {
#pragma unroll
            for (int c = 0; c < kUpChunks; ++c) {
                out[c]   = make_uint2(0, 0);
                scale[c] = __float2half(0.0F);
            }
            return;
        }
        const std::int64_t global_row = static_cast<std::int64_t>(local_row / columns) * kH +
                                        column_begin + local_row % columns;
        const std::int8_t* codes =
            p.up_codes + global_row * kUpKPad + up_half * (kR / 2) + 8 * quad;
        const __half* s = p.up_scales + global_row * kUpGroups + up_half * kUpChunks;
#pragma unroll
        for (int c = 0; c < kUpChunks; ++c) {
            out[c]   = load_streaming_u2(codes + 32 * c);
            scale[c] = s[c];
        }
    };

#pragma unroll
    for (int i = 0; i < kMaxTiles; ++i) {
        const int tile = slice + i * slices;
        if (tile < tiles) {
            float acc0 = 0.0F, acc1 = 0.0F;
#pragma unroll
            for (int c = 0; c < kDownChunks; ++c) {
                uint4 activation = make_uint4(0, 0, 0, 0);
                if (group < Tokens) {
                    activation = *reinterpret_cast<const uint4*>(
                        activations + group * kActStride + warp * kDownWarpK + 32 * c + 8 * quad);
                }
                float chunk[4] = {};
                mma_chunk(chunk, widen_codes(bits[i][c]), make_uint4(0, 0, 0, 0), activation);
                const float s = __half2float(scales[i][c]);
                acc0          = fmaf(chunk[0], s, acc0);
                acc1          = fmaf(chunk[1], s, acc1);
            }
            tile_shared[warp][group][2 * quad]     = acc0;
            tile_shared[warp][group][2 * quad + 1] = acc1;
        }
        load_up(i * 64 + up_tile * 16 + group, up_top[i], up_scale_top[i]);
        load_up(i * 64 + up_tile * 16 + group + 8, up_bottom[i], up_scale_bottom[i]);
        if (tile < tiles) {
            __syncthreads();
            if (tid < 64) {
                const int row   = tile * 8 + tid / 8;
                const int token = tid % 8;
                float sum       = 0.0F;
#pragma unroll
                for (int w = 0; w < kWarps; ++w) sum += tile_shared[w][tid / 8][token];
                if (row < rows && token < Tokens)
                    p.partials[(token * kHcRowsCap + row) * kLS + stream] = sum;
            }
            __syncthreads();
        }
    }

    constexpr int kMixItems = (kHcFusedMaxT * kMaxColumns + kThreads - 1) / kThreads; // 2
    float committed[kMixItems][kLS], mix_norm[kMixItems][kLS];
#pragma unroll
    for (int k = 0; k < kMixItems; ++k) {
        const int i = tid + k * kThreads;
        if (i >= Tokens * columns) continue;
        const int token = i / columns;
        const int d     = column_begin + i % columns;
        const float y =
            p.previous_y != nullptr
                ? __bfloat162float(p.previous_y[d + static_cast<std::int64_t>(kH) * token])
                : 0.0F;
#pragma unroll
        for (int s = 0; s < kLS; ++s) {
            const float h =
                __bfloat162float(p.residual[static_cast<std::int64_t>(kLW) * token + kH * s + d]);
            committed[k][s] = p.previous_y != nullptr
                                  ? __bfloat162float(__float2bfloat16_rn(
                                        h + p.previous_inject[s + kLS * token] * y))
                                  : h;
            mix_norm[k][s]  = 1.0F + __bfloat162float(p.norm[kH * s + d]);
        }
    }

    auto grid = cooperative_groups::this_grid();
    grid.sync();

    // Exchange 1: the projection's rank partial (the two local lanes summed in a fixed order),
    // p[t * rows + r] = fl32(p_0 + p_1).
    {
        const int count = Tokens * rows; // whole 16-byte vectors: rows * 4 bytes is one per token
        // Rank partial into p (in place), then exchanged and summed.
        for (int e = static_cast<int>(blockIdx.x) * kThreads + tid; e < count;
             e += static_cast<int>(gridDim.x) * kThreads) {
            const int token = e / rows;
            const int row   = e - token * rows;
            const float* q  = p.partials + (token * kHcRowsCap + row) * kLS;
            p.projection[e] = q[0] + q[1];
        }
    }
    grid.sync();
    grid_exchange(
        reinterpret_cast<const PeerVec*>(p.projection), p.exchange1, p.hang, Tokens * rows / 4,
        [&](int v, const float4& sum) { reinterpret_cast<float4*>(p.projection)[v] = sum; });
    grid.sync();

    // Phase 2.
    auto* low      = activations;
    float* up_sums = reinterpret_cast<float*>(activations + 8 * kLowStride);
    if (tid < Tokens * kLS)
        inverse_rms[tid] = rsqrtf(p.square_sums[tid] / static_cast<float>(kH) + p.eps);
    constexpr int kLowPairs     = Tokens * kR / 2;
    constexpr int kLowPerThread = (kLowPairs + kThreads - 1) / kThreads;
#pragma unroll
    for (int k = 0; k < kLowPerThread; ++k) {
        const int i = tid + k * kThreads;
        if (i < kLowPairs) {
            const int token = i / (kR / 2);
            const int rank  = 2 * (i % (kR / 2));
            const float v0  = p.projection[token * rows + rank] * (1.0F / kS);
            const float v1  = p.projection[token * rows + rank + 1] * (1.0F / kS);
            *reinterpret_cast<std::uint32_t*>(low + token * kLowStride + rank) =
                pack_f16(v0 * sigmoidf_(v0), v1 * sigmoidf_(v1));
        }
    }
    if (blockIdx.x == 0 && p.inject != nullptr && tid < Tokens * kLS) {
        const int token = tid / kLS;
        const float v = p.projection[token * rows + kR + p.first_stream + tid % kLS] * (1.0F / kS);
        p.inject[tid] = 2.0F * sigmoidf_(v);
    }
    __syncthreads();

    if (columns > 0) {
#pragma unroll
        for (int pass = 0; pass < kUpPasses; ++pass) {
            if (pass * 64 >= up_rows) break;
            const int top_row = pass * 64 + up_tile * 16 + group;
            float acc[4]      = {};
#pragma unroll
            for (int c = 0; c < kUpChunks; ++c) {
                uint4 activation = make_uint4(0, 0, 0, 0);
                if (group < Tokens) {
                    activation = *reinterpret_cast<const uint4*>(
                        low + group * kLowStride + up_half * (kR / 2) + 32 * c + 8 * quad);
                }
                float chunk[4] = {};
                mma_chunk(chunk, widen_codes(up_top[pass][c]), widen_codes(up_bottom[pass][c]),
                          activation);
                const float top    = __half2float(up_scale_top[pass][c]);
                const float bottom = __half2float(up_scale_bottom[pass][c]);
                acc[0]             = fmaf(chunk[0], top, acc[0]);
                acc[1]             = fmaf(chunk[1], top, acc[1]);
                acc[2]             = fmaf(chunk[2], bottom, acc[2]);
                acc[3]             = fmaf(chunk[3], bottom, acc[3]);
            }
            float* sums = up_sums + up_half * kMaxUpRows * 8;
            if (top_row < up_rows) {
                sums[top_row * 8 + 2 * quad]     = acc[0];
                sums[top_row * 8 + 2 * quad + 1] = acc[1];
            }
            if (top_row + 8 < up_rows) {
                sums[(top_row + 8) * 8 + 2 * quad]     = acc[2];
                sums[(top_row + 8) * 8 + 2 * quad + 1] = acc[3];
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kMixItems; ++k) {
            const int i = tid + k * kThreads;
            if (i >= Tokens * columns) continue;
            const int token  = i / columns;
            const int column = i % columns;
            const int d      = column_begin + column;
            float mixed      = 0.0F;
#pragma unroll
            for (int s = 0; s < kLS; ++s) {
                if (p.previous_y != nullptr)
                    p.residual[static_cast<std::int64_t>(kLW) * token + kH * s + d] =
                        __float2bfloat16_rn(committed[k][s]);
                const float normalized =
                    committed[k][s] * inverse_rms[token * kLS + s] * mix_norm[k][s];
                const int row     = (s * columns + column) * 8 + token;
                const float logit = up_sums[row] + up_sums[kMaxUpRows * 8 + row];
                mixed             = fmaf(normalized, sigmoidf_(logit), mixed);
            }
            p.collapse[d + static_cast<std::int64_t>(kH) * token] = mixed;
        }
    }
    grid.sync();

    // Exchange 2: mixed = RNE(fl32(q_0 + q_1) / S), rounded once.
    grid_exchange(reinterpret_cast<const PeerVec*>(p.collapse), p.exchange2, p.hang,
                  Tokens * kH / 4, [&](int v, const float4& sum) {
                      const float4 scaled = make_float4(sum.x * (1.0F / kS), sum.y * (1.0F / kS),
                                                        sum.z * (1.0F / kS), sum.w * (1.0F / kS));
                      reinterpret_cast<uint2*>(p.mixed)[v] = peer_pack_bf16x4(scaled);
                  });
}

std::size_t fused_shared_bytes(int tokens) {
    const std::size_t phase1 = static_cast<std::size_t>(tokens) * kActStride * 2;
    const std::size_t phase2 = 8 * kLowStride * 2 + 2 * kMaxUpRows * 8 * sizeof(float);
    return phase1 > phase2 ? phase1 : phase2;
}

using FusedKernel = void (*)(HcSplitFused);

template <int... Tokens>
constexpr auto fused_kernels(std::integer_sequence<int, Tokens...>) {
    return std::array<FusedKernel, sizeof...(Tokens)>{hc_split_fused_kernel<Tokens + 1>...};
}

constexpr auto kFusedKernels = fused_kernels(std::make_integer_sequence<int, kHcFusedMaxT>{});

int compute_grid() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int multiprocessors = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    int cooperative = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
    const int grid = multiprocessors / kLS * kLS;
    if (cooperative == 0 || grid < kMinGrid) return 0;
    for (int tokens = 1; tokens <= kHcFusedMaxT; ++tokens) {
        int resident = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, kFusedKernels[tokens - 1], kThreads, fused_shared_bytes(tokens)));
        if (resident == 0) return 0;
    }
    return grid;
}

} // namespace

int hyper_connection_split_fused_grid() {
    constexpr int kMaxDevices = 16;
    static int grids[kMaxDevices];
    static bool known[kMaxDevices];
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) return 0;
    if (!known[device]) {
        grids[device] = compute_grid();
        known[device] = true;
    }
    return grids[device];
}

std::size_t hyper_connection_split_fused_workspace_bytes(std::int32_t tokens) {
    const auto t = static_cast<std::size_t>(tokens);
    const auto a = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    return a(t * kHcRowsCap * kLS * 4) + a(t * kLS * 4) + a(t * kHcRowsCap * 4) + a(t * kH * 4);
}

void hyper_connection_split_fused_launch(const HcSplitFused& params, cudaStream_t stream) {
    const int grid = hyper_connection_split_fused_grid();
    cudaLaunchConfig_t config{};
    config.gridDim          = dim3(static_cast<unsigned>(grid));
    config.blockDim         = dim3(kThreads);
    config.dynamicSmemBytes = fused_shared_bytes(params.tokens);
    config.stream           = stream;
    cudaLaunchAttribute cooperative{};
    cooperative.id              = cudaLaunchAttributeCooperative;
    cooperative.val.cooperative = 1;
    config.attrs                = &cooperative;
    config.numAttrs             = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config, kFusedKernels[params.tokens - 1], params));
}

} // namespace ninfer::ops::detail
