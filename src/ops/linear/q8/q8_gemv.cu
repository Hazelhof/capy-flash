#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"

#include "ops/linear/q8/q8_output.cuh"

namespace ninfer::ops::detail {
void launch_q8_a16_gemv_r4_w1_k16384(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    q8_with_output(out, out.ne[0], [&](auto output) {
        launch_q8_a16_gemv<Q8A16GemvSchedule<4, 1, 2, 16384>>(q8_linear_operands(x, w), output, LinearIdentityEpilogue{}, stream);
    });
}
} // namespace ninfer::ops::detail
