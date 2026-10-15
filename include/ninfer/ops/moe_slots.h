#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Routed-expert MoE over an expert cache's device slots, at the Qwen3.8-Flash-Next geometry
// (hidden 2560, expert intermediate 640, top-10), for a decode window of 1..8 positions.

inline constexpr std::int32_t kMoeSlotsHidden       = 2560;
inline constexpr std::int32_t kMoeSlotsIntermediate = 640;
inline constexpr std::int32_t kMoeSlotsTopK         = 10;
inline constexpr std::int32_t kMoeSlotsMaxTokens    = 8;

// Bytes of one expert in each slot plane: the expert's row block of its layer bank in the
// `block_scale_k16_m128x4_v1` layout (models::qwen3_5::routed_expert_spans), without the
// divisor words.
inline constexpr std::int64_t kMoeSlotsGateUpCodeBytes  = 2LL * 640 * 2560 / 2;  // 1 638 400
inline constexpr std::int64_t kMoeSlotsGateUpScaleBytes = 2LL * 640 * 2560 / 16; //   204 800
inline constexpr std::int64_t kMoeSlotsDownCodeBytes    = 2560LL * 640 / 2;      //   819 200
inline constexpr std::int64_t kMoeSlotsDownScaleBytes   = 2560LL * 640 / 16;     //   102 400

// Plane q of slot s begins at base + s * stride (bytes). One array per plane and one contiguous
// slot holding the four planes back to back are both such descriptions.
struct MoeSlotPlane {
    const void* base    = nullptr;
    std::int64_t stride = 0;
};

// One GPU's slot pool (every layer's slots). A slot holds one routed expert:
//   gate_up_codes  [1280, 2560] E2M1, rows 0..639 gate, 640..1279 up (the bank's row order)
//   gate_up_scales the swizzled E4M3FN scale plane of those 1280 rows (10 whole 128-row tiles)
//   down_codes     [2560, 640] E2M1
//   down_scales    the swizzled E4M3FN scale plane of those 2560 rows
// Code planes need 16-byte aligned bases and strides, scale planes 4-byte.
struct MoeSlotPool {
    MoeSlotPlane gate_up_codes, gate_up_scales, down_codes, down_scales;
    std::int32_t slots = 0;
};

// One layer's view: which experts are resident where, and the per-expert scalars by expert id.
//   slot_table        I32 [E]: slot of routed expert e in the pool, or -1 (not on this GPU)
//   gate_up_divisors  FP32 [2E]: the layer gate|up bank's divisor plane (gate of e at 2e, up at
//                     2e+1), copied to the device once
//   down_divisors     FP32 [E]: the layer down bank's divisor plane
//   *_input_divisor   NVFP4 activation divisors of the layer's expert inputs x and h (the
//                     reciprocal of the calibrated input scale)
struct MoeSlotsLayer {
    Tensor slot_table;
    Tensor gate_up_divisors;
    Tensor down_divisors;
    float gate_up_input_divisor = 0.0F;
    float down_input_divisor    = 0.0F;
};

enum class MoeSlotsOutput : std::uint8_t {
    Store,      // out  = partial
    Accumulate, // out += partial (FP32 out only)
};

/**
 * Workspace bytes of moe_slots_nvfp4 for every T in [1, max_tokens].
 */
[[nodiscard]] std::size_t moe_slots_nvfp4_workspace_bytes(std::int32_t max_tokens);

/**
 * Op: NVFP4 (W4A4) routed-expert MoE partial over the experts resident in a slot pool
 *
 * Math / indexing, per token t < T and position k < 10, with e = routed_ids[k,t]:
 *   s(k,t) = slot_table[e] if 0 <= e < E and 0 <= slot_table[e] < pool.slots, else none
 *   W[n,c] = decode_e2m1(code[n,c]) * decode_e4m3fn(scale[n,c/16]) / divisor   (exact decode of
 *            slot s's planes; divisor = gate_up_divisors[2e] for gate rows, [2e+1] for up rows,
 *            down_divisors[e] for down)
 *   partial[:,t] = sum over k with s(k,t) of routed_weights[k,t] *
 *                  W_down(s) . ( silu(W_gate(s) x[:,t]) * (W_up(s) x[:,t]) )
 *   Store: out[:,t] = partial[:,t]; Accumulate: out[:,t] += partial[:,t].
 * Positions without a slot contribute nothing (missing experts are computed by the caller, e.g.
 * the CPU path); a token, or the whole call, without any resident expert gets 0 (Store) or keeps
 * `out` (Accumulate). An expert repeated in a token row counts with each of its weights.
 *
 * Logical shapes: x BF16 [2560,T]; routed_ids I32 [10,T]; routed_weights FP32 [10,T];
 *   out FP32 or BF16 [2560,T] (BF16 only with Store).
 * Supported domain: 1 <= T <= 8; E >= 1; pool.slots >= 1; finite positive input divisors;
 *   plane alignment as MoeSlotPool. Slot contents must be valid NVFP4 (finite scales).
 * Numeric (arithmetic profile W4A4, the FlashInfer/TensorRT-LLM NVFP4 MoE recipe): x and the
 *   SwiGLU output h are quantized to E2M1 with one E4M3FN scale per 16 values against the
 *   layer's input divisor; 16-value block products are exact integers scaled by both block
 *   scales; FP32 accumulation; gate/up outputs and h are rounded to BF16 before quantization;
 *   the partial is the FP32 sum over k in top-k order (deterministic, bit-reproducible, and
 *   independent of which slot holds an expert). FP64 oracle from represented inputs with
 *   unquantized activations.
 * Effects: writes `out` only; must not overlap any input or the workspace.
 * Workspace: moe_slots_nvfp4_workspace_bytes(T) bytes of `workspace`, no state between calls.
 * Execution: stream ordered, CUDA Graph capturable (table, slots and scalars are read at replay),
 *   both kernels launched as programmatic dependents (PDL) of the preceding kernel.
 */
void moe_slots_nvfp4(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                     const MoeSlotsLayer& layer, const MoeSlotPool& pool, MoeSlotsOutput mode,
                     Tensor& out, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
