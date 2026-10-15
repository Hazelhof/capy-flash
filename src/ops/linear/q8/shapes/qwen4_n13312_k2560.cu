#include "ops/linear/q8/shapes/qwen4_dense.cuh"

namespace ninfer::ops::detail {
// Qwen4Exp dense Q8 [13312, 2560] (qwen4_dense.cuh, T49).
Q8Launch select_q8_qwen4_n13312_k2560(std::int32_t tokens) {
    return qwen4_dense::select<Q8LinearGeometry<13312, 2560>>(tokens, 2); // T49 sweep (out/A/lin): candidate 2
}
} // namespace ninfer::ops::detail
