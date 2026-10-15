#include "ops/linear/q8/shapes/qwen4_dense.cuh"

namespace ninfer::ops::detail {
// Qwen4Exp dense Q8 [2560, 3072] (qwen4_dense.cuh, T49).
Q8Launch select_q8_qwen4_n2560_k3072(std::int32_t tokens) {
    return qwen4_dense::select<Q8LinearGeometry<2560, 3072>>(tokens, 2); // T49 sweep (out/A/lin): candidate 2
}
} // namespace ninfer::ops::detail
