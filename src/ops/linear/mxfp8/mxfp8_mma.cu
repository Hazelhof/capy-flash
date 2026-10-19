#include "ops/linear/mxfp8/mxfp8_mma.cuh"

namespace ninfer::ops::detail {

void launch_mxfp8_a16_mma(const Mxfp8LinearOperands& operands, Tensor& out, cudaStream_t stream) {
    if (out.dtype == DType::FP32) {
        launch_mxfp8_a16_mma_output(
            operands, LinearF32Output{static_cast<float*>(out.data), out.ne[0]}, stream);
    } else {
        launch_mxfp8_a16_mma_output(
            operands, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]}, stream);
    }
}

} // namespace ninfer::ops::detail
