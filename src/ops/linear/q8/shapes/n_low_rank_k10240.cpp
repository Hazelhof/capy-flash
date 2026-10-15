#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {

// Hyper-connection projections of the Qwen4Exp mixers ([320|324, 10240] onto the low rank,
// [10240, 320] back) and the PLE key/value parent [12800, 2560]. They are not multiples of the
// templated geometries, so they take the generic routes; T selection follows ninfer-ext
// (src/ops/linear/q8/shapes/qwen4_exp.cpp) and is untuned on the RTX 5070 Ti. The small-T
// hyper-connection mix does not use these routes (it contracts both projections in one fused
// launch); they serve the general route and the PLE projection.
Q8Launch select_q8_generic_hidden2560(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 16) return launch_q8_a16_simt_r8_t8;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    return launch_q8_a16_mma_r64_t128;
}

// Few output rows over a wide K: several warps share each row at decode widths.
Q8Launch select_q8_low_rank_k10240(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r4_t4_w2_g16_s2;
    if (tokens <= 16) return launch_q8_a16_sliced_r16_t16_w8_s2;
    return select_q8_generic_hidden2560(tokens);
}

} // namespace ninfer::ops::detail
