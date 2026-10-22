// Modified for capy-flash from fngine:src/kernels/moe/moe_decode.cu @ c20a885 (satellitedown/fngine, Apache License 2.0). See NOTICE.
//
// Implements: include/ninfer/ops/moe_slots_gguf.h (moe_slots_gguf)
//
// Decode window (T <= 8): hot-expert GEMVs over the native GGUF block formats, dp4a against q8
// activations (32-value blocks, FP32 scale). Kept from fngine: the integer sub-block decode, the
// persistent gate/up kernel streaming K slices of its tile rows through one shared buffer (the
// codebook staged in shared memory), the persistent down kernel with two cp.async stages and the
// per-item q8 quantization of h, and every per-(row, pair) FP32 summation order.
// Changes in this tree:
//   * fngine's router select + exchange publish kernel is replaced by a plan kernel over this
//     Op's inputs: routed_ids, the layer's slot table and either the blob offsets of the slot pool
//     or the planar bank bases; a work item is (slot, up to TMAX of its pairs), so a slot used by
//     more pairs than TMAX (an expert repeated in a token row) is split, never truncated;
//   * an item carries its gate|up and down row pointers instead of a slot index of a fixed-stride
//     blob array;
//   * hot_down writes the unweighted FP32 down output of each pair; a merge kernel sums the
//     positions of a token in top-k order (fmaf with the routed weight) into the FP32 (Store or
//     Accumulate) or BF16 (Store) output. fngine's CPU-tier combine, exchange and abliteration
//     variants are not carried over;
//   * the codebook is staged with 4-byte cp.async (the tables are only element-aligned) before
//     the PDL wait (it is not producer data);
//   * PDL through core/pdl.cuh; launch geometry cached per device.
#include "ops/moe_slots_gguf/iq_device.cuh"
#include "ops/moe_slots_gguf/launch.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Format = MoeGgufFormat;
using gguf_device::decode_sub32;
using gguf_device::kRowBytes;
using gguf_device::quantize_q8_lane;
using gguf_device::Sub32;
using gguf_device::sub32_dot;

constexpr int kHidden       = kMoeSlotsHidden;       // 2560
constexpr int kInter        = kMoeSlotsIntermediate; // 640
constexpr int kTopK         = kMoeSlotsTopK;         // 10
constexpr int kMaxPairs     = kGgufMaxPairs;         // 80
constexpr int kHiddenBlocks = kHidden / 32;          // 80 q8 blocks of x
constexpr int kInterBlocks  = kInter / 32;           // 20 q8 blocks of h

// PDL (fngine pdl.cuh): a const __restrict__ pointer compiles to ld.global.nc, which ptxas may
// schedule above griddepcontrol.wait. Producer-written data is read through coherent(pointer)
// (ordinary loads), and the few values hot_down reads ahead of its wait (written two kernels
// back, see hot_down_kernel) with gpu-scope loads that bypass L1.
template <class T>
__device__ __forceinline__ T* coherent(T* pointer) {
    asm("" : "+l"(pointer));
    return pointer;
}

__device__ __forceinline__ int load_settled(const int* p) {
    int v;
    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ const std::uint8_t* load_settled(const std::uint8_t* const* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return reinterpret_cast<const std::uint8_t*>(v);
}

// ------------------------------------------------------------------------------------ plan
// Block 0 maps every (token, position) to its slot and builds the work list; blocks 1..T quantize
// x[t] to q8 (32-value blocks, FP32 scale). Programmatic dependent of the preceding kernel: it
// triggers first (gate/up only stages its codebook before its own wait) and waits before reading
// any input (routing, table and offsets are read at replay).
//
// Pairs p = t * 10 + k with a slot are grouped by slot. Within a slot, pairs keep index order;
// rank r counts the slot's earlier pairs, and the pair at a rank that is a multiple of `cap`
// starts an item of min(cap, count - r) pairs. Items are numbered in the order of their first
// pair, and pair positions follow the items. Per-pair arithmetic does not depend on the grouping.
constexpr int kPlanThreads = 128;
static_assert(kMaxPairs <= kPlanThreads);

__global__ void __launch_bounds__(kPlanThreads)
    plan_kernel(const __nv_bfloat16* x, const std::int32_t* ids, int T, int cap, GgufExperts ex,
                GgufDecodeWork* work, std::int8_t* xq, float* xd) {
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (blockIdx.x > 0) {
        const int t                 = blockIdx.x - 1;
        const __nv_bfloat16* const xs = coherent(x) + std::size_t(t) * kHidden;
        for (int b = warp; b < kHiddenBlocks; b += kPlanThreads / 32) {
            const float v = __bfloat162float(xs[b * 32 + lane]);
            float d;
            const std::int8_t q                            = quantize_q8_lane(v, &d);
            xq[std::size_t(t) * kHidden + b * 32 + lane] = q;
            if (lane == 0) xd[t * kHiddenBlocks + b] = d;
        }
        return;
    }

    __shared__ int s_slot[kMaxPairs];
    __shared__ int s_rank[kMaxPairs];
    __shared__ int s_size[kMaxPairs];
    __shared__ int s_begin[kMaxPairs];
    const int p     = threadIdx.x;
    const int pairs = T * kTopK;
    if (p < kMaxPairs) {
        int s = -1;
        if (p < pairs) {
            const int e = coherent(ids)[p];
            if (e >= 0 && e < ex.experts) {
                const int v = coherent(ex.table)[e];
                s           = (v >= 0 && v < ex.slots) ? v : -1;
            }
        }
        s_slot[p] = s;
    }
    __syncthreads();
    int s = -1, rank = 0;
    if (p < kMaxPairs) {
        s         = s_slot[p];
        int count = 0;
        if (s >= 0) {
            for (int q = 0; q < pairs; ++q) {
                if (s_slot[q] == s) {
                    ++count;
                    if (q < p) ++rank;
                }
            }
        }
        s_rank[p] = rank;
        s_size[p] = (s >= 0 && rank % cap == 0) ? min(cap, count - rank) : 0;
    }
    __syncthreads();
    if (p < kMaxPairs && s_size[p] > 0) {
        int item = 0, begin = 0;
        for (int q = 0; q < p; ++q) {
            item += s_size[q] > 0 ? 1 : 0;
            begin += s_size[q];
        }
        s_begin[p]                  = begin;
        work->item_pair_begin[item] = begin;
        const std::uint8_t* gate_up;
        const std::uint8_t* down;
        if (ex.slot_offset != nullptr) {
            gate_up = ex.base + coherent(ex.slot_offset)[s];
            down    = gate_up + ex.gate_up_bytes;
        } else {
            gate_up = ex.gate_up_base + std::int64_t(s) * ex.gate_up_bytes;
            down    = ex.down_base + std::int64_t(s) * ex.down_bytes;
        }
        work->item_gate_up[item] = gate_up;
        work->item_down[item]    = down;
    }
    __syncthreads();
    if (p < pairs) {
        int pos = -1;
        if (s >= 0) {
            const int first = rank - rank % cap;
            for (int q = 0; q <= p; ++q) {
                if (s_slot[q] == s && s_rank[q] == first) {
                    pos = s_begin[q] + rank % cap;
                    break;
                }
            }
            work->pair_token[pos] = p / kTopK;
        }
        work->token_pair[p] = pos;
    }
    if (p == 0) {
        int n = 0, total = 0;
        for (int q = 0; q < kMaxPairs; ++q) {
            n += s_size[q] > 0 ? 1 : 0;
            total += s_size[q];
        }
        work->n_items            = n;
        work->item_pair_begin[n] = total;
    }
}

// ------------------------------------------------------------------------------- gate/up
constexpr int kGuThreads      = 128;                         // 4 warps x 2 rows
constexpr int kGuRowsPerTile  = kGuThreads / 32 * 2;         // 8
constexpr int kGuTilesPerItem = kInter / kGuRowsPerTile;     // 80
// A gate/up row is staged in kGuSlices K slices: slice i is row bytes [i * Row / 5, (i + 1) *
// Row / 5), whole quantization blocks holding sub-blocks [16 i, 16 i + 16), which is what a lane
// decodes in iteration i of the tile loop.
constexpr int kGuSlices = 5;
static_assert(kHiddenBlocks == 16 * kGuSlices, "one slice per 16 sub-blocks of a hidden row");

__host__ __device__ constexpr std::size_t align16(std::size_t v) { return (v + 15) & ~std::size_t{15}; }

// Shared-memory stride for one staged weight row of `bytes` (copied from the 16-byte aligned
// address at or below its start).
__host__ __device__ constexpr int staged_stride(int bytes) { return (bytes + 15 + 15) / 16 * 16; }

__device__ __forceinline__ int staged_offset(const std::uint8_t* row) {
    return int(reinterpret_cast<std::uintptr_t>(row) & 15);
}

// Warp-cooperative copy of one weight row into shared memory with 16-byte cp.async (L1 bypass);
// bytes at or past `limit` are zero-filled. The row starts staged_offset(row) bytes into `dst`.
template <int Stride>
__device__ __forceinline__ void warp_stage_row(std::uint8_t* dst, const std::uint8_t* row,
                                               const std::uint8_t* limit, int lane) {
    const std::uint8_t* base = row - staged_offset(row);
    for (int ch = lane; ch < Stride / 16; ch += 32) {
        const std::uint8_t* src = base + ch * 16;
        const std::int64_t left = limit - src;
        cp_async_zfill<16, Cache::cg>(dst + ch * 16, src,
                                      left >= 16 ? 16 : (left > 0 ? int(left) : 0));
    }
}

// Warp-cooperative cp.async of slice i of one staged row (the buffer layout of warp_stage_row):
// the 16-byte chunks whose first row byte lies in the slice (chunk 0 always joins slice 0), so
// slices 0..i of a row are complete once their copies are. Whatever the row's offset, slice i's
// chunks lie below staged byte (i + 1) * S + 31 (S = Row / 5) and the chunks holding bytes of slice
// i + 2 start at or above (i + 2) * S - 15: hot_gate_up refills slice i of the next tile as soon as
// slice i + 1 of the current one is decoded.
template <int Row>
__device__ __forceinline__ void warp_stage_slice(std::uint8_t* dst, const std::uint8_t* row,
                                                 const std::uint8_t* limit, int i, int lane) {
    constexpr int kSliceBytes = Row / kGuSlices;
    static_assert(Row % kGuSlices == 0, "slices hold whole quantization blocks");
    static_assert(kSliceBytes >= 46, "a refilled slice i must end before the chunks of slice i + 2");
    const int off            = staged_offset(row);
    const std::uint8_t* base = row - off;
    const int lo             = i == 0 ? 0 : (i * kSliceBytes + off + 15) / 16;
    const int hi             = ((i + 1) * kSliceBytes + off + 15) / 16;
    for (int ch = lo + lane; ch < hi; ch += 32) {
        const std::uint8_t* src = base + ch * 16;
        const std::int64_t left = limit - src;
        cp_async_zfill<16, Cache::cg>(dst + ch * 16, src,
                                      left >= 16 ? 16 : (left > 0 ? int(left) : 0));
    }
}

// q8 activation sub-block (32 int8 as 8 words) of token t.
__device__ __forceinline__ void load_q8(const std::int8_t* xq, int t, int sub, int K, int (&u)[8]) {
    const int4* p  = reinterpret_cast<const int4*>(xq + std::size_t(t) * K + sub * 32);
    const int4 u0 = p[0], u1 = p[1];
    u[0] = u0.x;
    u[1] = u0.y;
    u[2] = u0.z;
    u[3] = u0.w;
    u[4] = u1.x;
    u[5] = u1.y;
    u[6] = u1.z;
    u[7] = u1.w;
}

// Persistent grid (as many CTAs as fit on every SM) over tiles (work item, 8 intermediate rows),
// tile index item * 80 + row group; CTA c takes tiles c, c + G, ... Each warp stages the gate and
// up rows of its two rows (coalesced 16-byte copies; the CTA stages the format's codebook once),
// then half-warps own one row: each of 16 lanes covers 5 of the 80 sub-blocks, one per K slice;
// weights are decoded once and applied to every pair of the item. A single tile buffer streams:
// once slice i of the current tile is decoded, slice i - 1 of the next tile is requested in its
// place. Per lane the FP32 order is that of one tile per CTA.
// Programmatic dependent of the plan kernel: everything it reads comes from the plan, so it
// waits first, and only then lets hot_down launch, because hot_down reads the work list before its
// own wait.
template <Format F, int TMAX>
__global__ void __launch_bounds__(kGuThreads)
    hot_gate_up_kernel(const GgufDecodeWork* work, const std::int8_t* xq, const float* xd,
                       float* h) {
    constexpr int kGrid = gguf_device::kGridBytes<F>;
    __shared__ __align__(16) std::uint8_t s_grid[kGrid > 0 ? kGrid : 16];
    if constexpr (kGrid > 0) {
        // 4-byte copies: the ggml tables are uint32/uint64 arrays, aligned to their element only.
        const auto* grid = static_cast<const std::uint8_t*>(gguf_device::grid_table<F>());
        for (int c = threadIdx.x; c < kGrid / 4; c += kGuThreads)
            cp_async<4>(s_grid + 4 * c, grid + 4 * c);
    }
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    constexpr int kItems  = TMAX * kTopK; // work items and pairs of a call
    constexpr int kRow    = kRowBytes<F, kHidden>;
    constexpr int kStride = staged_stride(kRow);
    __shared__ __align__(16) std::uint8_t s_raw[kGuThreads / 32][4][kStride];
    __shared__ const std::uint8_t* s_rows[kItems]; // item -> gate|up rows
    __shared__ int s_tok[kItems];                  // pair -> token
    __shared__ int s_begin[kItems + 1];            // item -> first pair

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int half = lane >> 4, l16 = lane & 15;
    // One dependent round trip (the work list; entries past n_items are stale but in bounds)
    // precedes the weight staging.
    const GgufDecodeWork* const wk = coherent(work);
    const int n_tiles              = wk->n_items * kGuTilesPerItem;
    for (int i = threadIdx.x; i <= kItems; i += kGuThreads) {
        if (i < kItems) {
            s_rows[i] = wk->item_gate_up[i];
            s_tok[i]  = wk->pair_token[i];
        }
        s_begin[i] = wk->item_pair_begin[i];
    }
    __syncthreads();

    const int stride = int(gridDim.x);
    // Staged row s of tile t: gate (s even) or up (s odd) row r0 + s / 2 of the warp.
    auto row_of = [&](int t, int s) {
        const int r = (t % kGuTilesPerItem) * kGuRowsPerTile + warp * 2 + (s >> 1);
        return s_rows[t / kGuTilesPerItem] + ((s & 1) ? std::int64_t(kInter) * kRow : 0) +
               std::int64_t(r) * kRow;
    };
    // Requests slice i of the warp's four rows of tile t; always commits one group, so that a
    // tile's slices are groups in order.
    auto stage_slice = [&](int t, int i) {
        if (t < n_tiles) {
            const std::uint8_t* limit = s_rows[t / kGuTilesPerItem] + std::int64_t(2 * kInter) * kRow;
#pragma unroll
            for (int s = 0; s < 4; ++s)
                warp_stage_slice<kRow>(s_raw[warp][s], row_of(t, s), limit, i, lane);
        }
        cp_commit();
    };
    int t = blockIdx.x;
#pragma unroll
    for (int i = 0; i < kGuSlices; ++i) stage_slice(t, i); // group 0 also holds the codebook

    const std::int8_t* const xqs = coherent(xq);
    const float* const xds       = coherent(xd);
    for (bool first = true; t < n_tiles; first = false, t += stride) {
        const int item = t / kGuTilesPerItem;
        const int pb   = s_begin[item];
        const int n    = s_begin[item + 1] - pb;
        const int r    = (t % kGuTilesPerItem) * kGuRowsPerTile + warp * 2 + half;
        const std::uint8_t* grow = s_raw[warp][2 * half] + staged_offset(row_of(t, 2 * half));
        const std::uint8_t* urow =
            s_raw[warp][2 * half + 1] + staged_offset(row_of(t, 2 * half + 1));
        float ag[TMAX], au[TMAX];
#pragma unroll
        for (int j = 0; j < TMAX; ++j) ag[j] = au[j] = 0.f;
#pragma unroll
        for (int i = 0; i < kGuSlices; ++i) {
            // Groups committed after slice i of this tile: its later slices, plus slices
            // 0..i - 2 of the next tile requested so far.
            if (i == 0) cp_wait<kGuSlices - 1>();
            else cp_wait<kGuSlices - 2>();
            if (first && i == 0) __syncthreads(); // the codebook was staged by the whole CTA
            else __syncwarp();                    // otherwise a warp reads only the rows it staged
            const int sub = l16 + 16 * i;
            Sub32 sg, su;
            decode_sub32<F>(grow, sub, sg, s_grid);
            decode_sub32<F>(urow, sub, su, s_grid);
#pragma unroll
            for (int j = 0; j < TMAX; ++j) {
                if (j < n) {
                    const int tok = s_tok[pb + j];
                    int u[8];
                    load_q8(xqs, tok, sub, kHidden, u);
                    const float dx = xds[tok * kHiddenBlocks + sub];
                    ag[j] += sg.d * dx * float(sub32_dot(sg, u));
                    au[j] += su.d * dx * float(sub32_dot(su, u));
                }
            }
            if (i > 0) {
                __syncwarp(); // every lane is done with slices i - 1 and i of this tile
                stage_slice(t + stride, i - 1);
            }
        }
        stage_slice(t + stride, kGuSlices - 1);
#pragma unroll
        for (int j = 0; j < TMAX; ++j) {
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                ag[j] += __shfl_xor_sync(0xffffffffu, ag[j], o);
                au[j] += __shfl_xor_sync(0xffffffffu, au[j], o);
            }
        }
        if (l16 == 0) {
#pragma unroll
            for (int j = 0; j < TMAX; ++j)
                if (j < n) h[std::size_t(pb + j) * kInter + r] = silu(ag[j]) * au[j];
        }
    }
    cp_wait<0>();
}

// --------------------------------------------------------------------------------- down
// hot_down runs a persistent grid over fixed-size tiles and keeps kStages tiles staged per CTA
// (cp.async double buffering): the next tile's weights stream in while the current one is
// decoded. A tile's arithmetic, and therefore every FP32 summation order, is the same as with one
// tile per CTA.
constexpr int kStages           = 2;
constexpr int kDownThreads      = 128;                          // 4 warps x 8 rows
constexpr int kDownRowsPerTile  = kDownThreads / 32 * 8;        // 32
constexpr int kDownTilesPerItem = kHidden / kDownRowsPerTile;   // 80

// Dynamic shared memory of hot_down_kernel<F, TMAX>: kStages x [4 warps] staged 8-row ranges, the
// q8 h blocks of one item's pairs, and the work list's item rows and pair ranges.
template <Format F, int TMAX>
struct DownSmem {
    static constexpr int kItems      = TMAX * kTopK;
    static constexpr int kStride     = staged_stride(8 * kRowBytes<F, kInter>);
    static constexpr std::size_t kRaw = std::size_t(kStages) * (kDownThreads / 32) * kStride;
    static constexpr std::size_t kQ   = std::size_t(TMAX) * kInterBlocks * 32; // int8 blocks
    static constexpr std::size_t kD   = align16(sizeof(float) * TMAX * kInterBlocks);
    static constexpr std::size_t kP   = align16(sizeof(void*) * kItems);
    static constexpr std::size_t kBytes = kRaw + kQ + kD + kP + align16(sizeof(int) * (kItems + 1));
};

// Tile = (work item, 32 output rows), tile index item * 80 + row group; CTA c takes a contiguous
// tile range, so it meets one or two items and quantizes each one's h to q8 once. Each warp stages
// its 8 consecutive weight rows (one contiguous byte range); quads of lanes own one row (5 of the
// 20 sub-blocks each). Programmatic dependent of hot_gate_up: the first tiles' weight staging
// (independent of h) starts while gate/up finishes, and the wait orders the h reads after it. The
// work list is read before the wait: hot_gate_up triggers only after its own wait, so the plan
// kernel had completed before this grid could launch.
template <Format F, int TMAX>
__global__ void __launch_bounds__(kDownThreads)
    hot_down_kernel(const GgufDecodeWork* work, const float* h, float* partial) {
    pdl::trigger_dependents();
    using S            = DownSmem<F, TMAX>;
    constexpr int kRow = kRowBytes<F, kInter>;
    extern __shared__ __align__(16) std::uint8_t smem[];
    std::int8_t* const s_q = reinterpret_cast<std::int8_t*>(smem + S::kRaw);     // [TMAX][20][32]
    float* const s_d       = reinterpret_cast<float*>(smem + S::kRaw + S::kQ);   // [TMAX][20]
    const std::uint8_t** const s_mat =
        reinterpret_cast<const std::uint8_t**>(smem + S::kRaw + S::kQ + S::kD);
    int* const s_begin = reinterpret_cast<int*>(smem + S::kRaw + S::kQ + S::kD + S::kP);

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n_tiles = load_settled(&work->n_items) * kDownTilesPerItem;
    for (int i = threadIdx.x; i <= S::kItems; i += kDownThreads) {
        if (i < S::kItems) s_mat[i] = load_settled(&work->item_down[i]);
        s_begin[i] = load_settled(&work->item_pair_begin[i]);
    }
    __syncthreads();
    const int t_begin = int(std::int64_t(blockIdx.x) * n_tiles / int(gridDim.x));
    const int t_end   = int(std::int64_t(blockIdx.x + 1) * n_tiles / int(gridDim.x));

    auto first_row  = [&](int t) { return (t % kDownTilesPerItem) * kDownRowsPerTile + warp * 8; };
    auto stage_tile = [&](int t, int stage) {
        if (t < t_end) {
            const std::uint8_t* mat = s_mat[t / kDownTilesPerItem];
            warp_stage_row<S::kStride>(
                smem + (std::size_t(stage) * (kDownThreads / 32) + warp) * S::kStride,
                mat + std::int64_t(first_row(t)) * kRow, mat + std::int64_t(kHidden) * kRow, lane);
        }
        cp_commit();
    };
#pragma unroll
    for (int i = 0; i < kStages; ++i) stage_tile(t_begin + i, i);
    pdl::wait_for_dependencies();

    const float* const hs = coherent(h); // hot_gate_up's output
    constexpr int kWarps  = kDownThreads / 32;
    constexpr int kTasksPerWarp = (TMAX * kInterBlocks + kWarps - 1) / kWarps;
    constexpr int kBatch        = kTasksPerWarp < 8 ? kTasksPerWarp : 8;
    const int g4 = lane >> 2, l4 = lane & 3;
    int q_item   = -1; // item whose q8 h blocks are in s_q / s_d
    for (int t = t_begin, k = 0; t < t_end; ++t, ++k) {
        const int stage = k % kStages; // the loop refills the stage it just consumed
        const int item  = t / kDownTilesPerItem;
        const int pb    = s_begin[item];
        const int n     = s_begin[item + 1] - pb;
        if (item != q_item) {
            // q8 blocks of the item's h rows: block task = j * kInterBlocks + b of pair pb + j. A
            // warp issues the loads of a whole batch before quantizing it.
            if (q_item >= 0) __syncthreads(); // every warp is done with the previous item's blocks
            const float* const hp = hs + std::size_t(pb) * kInter + lane;
            const int n_tasks     = n * kInterBlocks;
            for (int t0 = warp; t0 < n_tasks; t0 += kWarps * kBatch) {
                float v[kBatch];
#pragma unroll
                for (int b = 0; b < kBatch; ++b) {
                    const int task = t0 + b * kWarps;
                    v[b]           = task < n_tasks ? hp[task * 32] : 0.f;
                }
#pragma unroll
                for (int b = 0; b < kBatch; ++b) {
                    const int task = t0 + b * kWarps;
                    if (task < n_tasks) { // warp-uniform: quantize_q8_lane reduces over the warp
                        float d;
                        s_q[task * 32 + lane] = quantize_q8_lane(v[b], &d);
                        if (lane == 0) s_d[task] = d;
                    }
                }
            }
            q_item = item;
            __syncthreads();
        }
        cp_wait<kStages - 1>();
        __syncwarp(); // a warp reads only the rows it staged

        const int r = first_row(t) + g4;
        const std::uint8_t* row =
            smem + (std::size_t(stage) * (kDownThreads / 32) + warp) * S::kStride +
            staged_offset(s_mat[item] + std::int64_t(first_row(t)) * kRow) + g4 * kRow;
        float acc[TMAX];
#pragma unroll
        for (int j = 0; j < TMAX; ++j) acc[j] = 0.f;
#pragma unroll
        for (int i = 0; i < kInterBlocks / 4; ++i) {
            const int sub = l4 + 4 * i;
            Sub32 s;
            decode_sub32<F>(row, sub, s, gguf_device::grid_table<F>());
#pragma unroll
            for (int j = 0; j < TMAX; ++j) {
                if (j < n) {
                    const int4* q =
                        reinterpret_cast<const int4*>(s_q + (j * kInterBlocks + sub) * 32);
                    const int4 u0 = q[0], u1 = q[1];
                    const int u[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
                    acc[j] += s.d * s_d[j * kInterBlocks + sub] * float(sub32_dot(s, u));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < TMAX; ++j) {
            acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 2);
            acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 1);
        }
        if (l4 == 0) {
#pragma unroll
            for (int j = 0; j < TMAX; ++j)
                if (j < n) partial[std::size_t(pb + j) * kHidden + r] = acc[j];
        }
        __syncwarp(); // every lane is done with this stage before it is refilled
        stage_tile(t + kStages, stage);
    }
    cp_wait<0>();
}

// ------------------------------------------------------------------------------- merge
// One CTA per token; thread i owns features [8i, 8i + 8). The positions of the token are summed
// in top-k order (fmaf with the routed weight); positions without a slot add nothing.
constexpr int kMergeThreads = kHidden / 8; // 320

__global__ void __launch_bounds__(kMergeThreads)
    merge_kernel(const GgufDecodeWork* work, const float* weights, const float* partial,
                 int accumulate, float* out_f32, __nv_bfloat16* out_bf16) {
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    const int t                    = blockIdx.x;
    const int f                    = threadIdx.x * 8;
    const GgufDecodeWork* const wk = coherent(work);
    const float* const w           = coherent(weights);
    const float* const ps          = coherent(partial);
    float sum[8]                   = {};
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        const int pos = wk->token_pair[t * kTopK + k];
        if (pos < 0) continue;
        const float weight = w[t * kTopK + k];
        const float* src   = ps + std::size_t(pos) * kHidden + f;
        const float4 a     = *reinterpret_cast<const float4*>(src);
        const float4 b     = *reinterpret_cast<const float4*>(src + 4);
        const float v[8]   = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) sum[i] = fmaf(weight, v[i], sum[i]);
    }
    const std::size_t o = std::size_t(t) * kHidden + f;
    if (out_f32 != nullptr) {
        float* const dst = coherent(out_f32) + o;
        if (accumulate) {
            const float4 a = *reinterpret_cast<const float4*>(dst);
            const float4 b = *reinterpret_cast<const float4*>(dst + 4);
            const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[i] = v[i] + sum[i];
        }
        *reinterpret_cast<float4*>(dst)     = make_float4(sum[0], sum[1], sum[2], sum[3]);
        *reinterpret_cast<float4*>(dst + 4) = make_float4(sum[4], sum[5], sum[6], sum[7]);
    } else {
        uint4 packed;
        packed.x = pack_bf16x2(sum[0], sum[1]);
        packed.y = pack_bf16x2(sum[2], sum[3]);
        packed.z = pack_bf16x2(sum[4], sum[5]);
        packed.w = pack_bf16x2(sum[6], sum[7]);
        *reinterpret_cast<uint4*>(out_bf16 + o) = packed;
    }
}

// ------------------------------------------------------------------------------ dispatch
int tmax_for(int tokens) {
    if (tokens <= 1) return 1;
    if (tokens <= 2) return 2;
    if (tokens <= 4) return 4;
    return 8;
}

// Resident CTAs of `Kernel` on the current device (all SMs), computed once per device; the
// dynamic shared-memory limit is raised first when the layout needs it.
constexpr int kMaxDevices = 16;

template <auto Kernel>
int resident_ctas(int threads, std::size_t smem) {
    static int cache[kMaxDevices] = {};
    int device                    = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices)
        throw std::runtime_error("moe_slots_gguf: device index out of range");
    if (cache[device] == 0) {
        if (smem > 48 * 1024)
            CUDA_CHECK(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        int sms = 0, per_sm = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, Kernel, threads, smem));
        if (per_sm < 1) throw std::runtime_error("moe_slots_gguf: kernel does not fit on an SM");
        cache[device] = sms * per_sm;
    }
    return cache[device];
}

template <Format F, int TMAX>
void launch_gate_up(int tokens, const GgufDecodeWorkspace& ws, cudaStream_t stream) {
    constexpr auto kernel = &hot_gate_up_kernel<F, TMAX>;
    const int ctas =
        std::min(resident_ctas<kernel>(kGuThreads, 0), tokens * kTopK * kGuTilesPerItem);
    CUDA_CHECK(pdl::launch_dependent(pdl::LaunchConfig{dim3(ctas), dim3(kGuThreads), 0, stream},
                                     kernel, static_cast<const GgufDecodeWork*>(ws.work),
                                     static_cast<const std::int8_t*>(ws.xq),
                                     static_cast<const float*>(ws.xd), ws.h));
}

template <Format F, int TMAX>
void launch_down(int tokens, const GgufDecodeWorkspace& ws, cudaStream_t stream) {
    constexpr auto kernel      = &hot_down_kernel<F, TMAX>;
    constexpr std::size_t smem = DownSmem<F, TMAX>::kBytes;
    const int ctas =
        std::min(resident_ctas<kernel>(kDownThreads, smem), tokens * kTopK * kDownTilesPerItem);
    CUDA_CHECK(pdl::launch_dependent(pdl::LaunchConfig{dim3(ctas), dim3(kDownThreads), smem, stream},
                                     kernel, static_cast<const GgufDecodeWork*>(ws.work),
                                     static_cast<const float*>(ws.h), ws.partial));
}

template <int TMAX>
void gate_up_for(Format format, int tokens, const GgufDecodeWorkspace& ws, cudaStream_t stream) {
    switch (format) {
    case Format::IQ2_S: return launch_gate_up<Format::IQ2_S, TMAX>(tokens, ws, stream);
    case Format::IQ3_XXS: return launch_gate_up<Format::IQ3_XXS, TMAX>(tokens, ws, stream);
    case Format::IQ3_S: return launch_gate_up<Format::IQ3_S, TMAX>(tokens, ws, stream);
    case Format::IQ4_XS: return launch_gate_up<Format::IQ4_XS, TMAX>(tokens, ws, stream);
    case Format::IQ4_NL: return launch_gate_up<Format::IQ4_NL, TMAX>(tokens, ws, stream);
    default: throw std::invalid_argument("moe_slots_gguf: unsupported gate/up format");
    }
}

template <int TMAX>
void down_for(Format format, int tokens, const GgufDecodeWorkspace& ws, cudaStream_t stream) {
    switch (format) {
    case Format::IQ4_NL: return launch_down<Format::IQ4_NL, TMAX>(tokens, ws, stream);
    case Format::Q2_0: return launch_down<Format::Q2_0, TMAX>(tokens, ws, stream);
    default: throw std::invalid_argument("moe_slots_gguf: unsupported down format");
    }
}

std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

} // namespace

std::size_t gguf_decode_workspace_layout(std::int32_t tokens, std::size_t offsets[5]) {
    const auto t            = static_cast<std::size_t>(tokens);
    const std::size_t pairs = t * kTopK;
    std::size_t at          = 0;
    offsets[0]              = at;
    at += align256(sizeof(GgufDecodeWork));
    offsets[1] = at;
    at += align256(t * kHidden);
    offsets[2] = at;
    at += align256(t * kHiddenBlocks * sizeof(float));
    offsets[3] = at;
    at += align256(pairs * kInter * sizeof(float));
    offsets[4] = at;
    at += align256(pairs * kHidden * sizeof(float));
    return at;
}

void gguf_decode_launch(const GgufDecodeCall& c, const GgufDecodeWorkspace& ws,
                        cudaStream_t stream) {
    const int T    = c.tokens;
    const int tmax = tmax_for(T);
    CUDA_CHECK(pdl::launch_dependent(
        pdl::LaunchConfig{dim3(1 + T), dim3(kPlanThreads), 0, stream}, plan_kernel, c.x, c.ids, T,
        tmax, c.experts, ws.work, ws.xq, ws.xd));
    switch (tmax) {
    case 1: gate_up_for<1>(c.experts.gate_up_format, T, ws, stream); break;
    case 2: gate_up_for<2>(c.experts.gate_up_format, T, ws, stream); break;
    case 4: gate_up_for<4>(c.experts.gate_up_format, T, ws, stream); break;
    default: gate_up_for<8>(c.experts.gate_up_format, T, ws, stream); break;
    }
    switch (tmax) {
    case 1: down_for<1>(c.experts.down_format, T, ws, stream); break;
    case 2: down_for<2>(c.experts.down_format, T, ws, stream); break;
    case 4: down_for<4>(c.experts.down_format, T, ws, stream); break;
    default: down_for<8>(c.experts.down_format, T, ws, stream); break;
    }
    CUDA_CHECK(pdl::launch_dependent(
        pdl::LaunchConfig{dim3(T), dim3(kMergeThreads), 0, stream}, merge_kernel,
        static_cast<const GgufDecodeWork*>(ws.work), c.weights,
        static_cast<const float*>(ws.partial), c.accumulate ? 1 : 0, c.out_f32, c.out_bf16));
}

} // namespace ninfer::ops::detail
