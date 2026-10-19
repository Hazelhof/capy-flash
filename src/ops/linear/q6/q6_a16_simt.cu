#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q6/q6_simt_launch.cuh"
#include "ops/linear/q6/q6_instances.cuh"

namespace ninfer::ops::detail {

void launch_q6_a16_simt_r8_t4(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    launch_q6_a16_simt<q6_instances::SimtR8T4>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_gemv_r4_w2_g16(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    launch_q6_a16_gemv<q6_instances::GemvR4W2G16>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

#define NINFER_Q6_GEMV_INSTANCE(name, schedule)                                                    \
    void name(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {           \
        launch_q6_a16_gemv<q6_instances::schedule>(                                                \
            q6_linear_operands(x, weight),                                                         \
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},                     \
            LinearIdentityEpilogue{}, stream);                                                     \
    }
NINFER_Q6_GEMV_INSTANCE(launch_q6_a16_gemv_r4_w2_g20_cg, GemvR4W2G20Cg)
NINFER_Q6_GEMV_INSTANCE(launch_q6_a16_gemv_r8_w1_g8_s4_cg, GemvR8W1G8S4Cg)
NINFER_Q6_GEMV_INSTANCE(launch_q6_a16_gemv_r4_w2_g12_s3_cg, GemvR4W2G12S3Cg)
NINFER_Q6_GEMV_INSTANCE(launch_q6_a16_gemv_r2_w4_g12_cg, GemvR2W4G12Cg)
#undef NINFER_Q6_GEMV_INSTANCE

} // namespace ninfer::ops::detail
