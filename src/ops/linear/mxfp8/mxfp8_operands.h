#pragma once

// MXFP8_E4M3FN_G32_UE8M0 in layout BlockScaleK32: an N x K plane of E4M3FN codes (row-major,
// K bytes per row) and an N x K/32 plane of UE8M0 exponents (row-major, K/32 bytes per row).
// The logical weight is e4m3(code) * 2^(scale - 127), exact in FP32.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Mxfp8LinearOperands {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    std::int32_t rows;
    std::int32_t k;
    std::int32_t tokens;
};

inline Mxfp8LinearOperands mxfp8_linear_operands(const Tensor& x, const Weight& weight) {
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            weight.n,
            weight.k,
            x.ne[1]};
}

// Small-T route (T <= kMxfp8GemvMaxTokens): one warp streams whole rows, FP32 dot products.
// From T = 6 the one-warp 16 x 8 Tensor Core tile streams the weight faster on the problems with
// N >= 2560 rows (measured on the RTX 5070 Ti, e.g. [2560,6144] T = 8: 35 vs 68 us), so those
// take the MMA route; the narrow ones (N = 48, 96, 640, 1280) keep the GEMV up to T = 8.
inline constexpr std::int32_t kMxfp8GemvMaxTokens = 8;

constexpr bool mxfp8_uses_gemv(std::int32_t rows, std::int32_t tokens) {
    return tokens <= 5 || (tokens <= kMxfp8GemvMaxTokens && rows < 2560);
}

void launch_mxfp8_a16_gemv(const Mxfp8LinearOperands& operands, Tensor& out, cudaStream_t stream);
void launch_mxfp8_a16_mma(const Mxfp8LinearOperands& operands, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
