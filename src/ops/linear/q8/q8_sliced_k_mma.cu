#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_launch.h"

#include "ops/linear/q8/q8_output.cuh"

namespace ninfer::ops::detail {
void launch_q8_a16_sliced_r16_t16_w8_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    q8_with_output(out, out.ne[0], [&](auto output) {
        launch_q8_a16_sliced_k_mma<q8_instances::SlicedR16T16W8S2>(q8_linear_operands(x, w), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q8_a16_sliced_r16_t16_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    q8_with_output(out, out.ne[0], [&](auto output) {
        launch_q8_a16_sliced_k_mma<q8_instances::SlicedR16T16W4S2>(q8_linear_operands(x, w), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q8_a16_sliced_r16_t32_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    q8_with_output(out, out.ne[0], [&](auto output) {
        launch_q8_a16_sliced_k_mma<q8_instances::SlicedR16T32W4S2>(q8_linear_operands(x, w), output, LinearIdentityEpilogue{}, stream);
    });
}
} // namespace ninfer::ops::detail
