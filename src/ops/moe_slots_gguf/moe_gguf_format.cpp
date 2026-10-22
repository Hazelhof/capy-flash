// Geometry helpers of the GGUF routed-expert Ops (include/ninfer/ops/moe_slots_gguf.h).
#include "ninfer/ops/moe_slots_gguf.h"

namespace ninfer::ops {
namespace {

struct Block {
    std::int32_t elements = 0, bytes = 0;
};

Block block_of(MoeGgufFormat format) noexcept {
    switch (format) {
    case MoeGgufFormat::IQ3_XXS: return {256, 98};
    case MoeGgufFormat::IQ4_NL: return {32, 18};
    case MoeGgufFormat::IQ3_S: return {256, 110};
    case MoeGgufFormat::IQ2_S: return {256, 82};
    case MoeGgufFormat::IQ4_XS: return {256, 136};
    case MoeGgufFormat::Q2_0: return {64, 18};
    }
    return {};
}

} // namespace

std::int64_t moe_gguf_row_bytes(MoeGgufFormat format, std::int32_t k) noexcept {
    const Block b = block_of(format);
    if (b.elements == 0 || k <= 0 || k % b.elements) { return 0; }
    return static_cast<std::int64_t>(k / b.elements) * b.bytes;
}

bool moe_gguf_gate_up_supported(MoeGgufFormat format) noexcept {
    switch (format) {
    case MoeGgufFormat::IQ2_S:
    case MoeGgufFormat::IQ3_XXS:
    case MoeGgufFormat::IQ3_S:
    case MoeGgufFormat::IQ4_XS:
    case MoeGgufFormat::IQ4_NL: return true;
    default: return false;
    }
}

bool moe_gguf_down_supported(MoeGgufFormat format) noexcept {
    return format == MoeGgufFormat::IQ4_NL || format == MoeGgufFormat::Q2_0;
}

std::int64_t moe_gguf_expert_bytes(MoeGgufFormat gate_up, MoeGgufFormat down) noexcept {
    return 1280 * moe_gguf_row_bytes(gate_up, 2560) + 2560 * moe_gguf_row_bytes(down, 640);
}

} // namespace ninfer::ops
