#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

// Qwen4Exp output head [248320, 2560] (Q6 in the `qwen3_8_flash_next_nvfp4` recipe). T selection
// is ninfer-ext's (src/ops/linear/q6/shapes/n248320_k2560.cpp, Apache-2.0), untuned on the
// RTX 5070 Ti.
Q6Launch select_q6_n248320_k2560(std::int32_t tokens) {
    if (tokens <= 2) return launch_q6_a16_simt_r8_t4;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q6_a16_mma_r64_t64_k128;
    if (tokens <= 96) return launch_q6_a16_mma_r64_t96;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
