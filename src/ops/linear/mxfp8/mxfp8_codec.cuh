#pragma once

// Exact decoders of the MXFP8 operands shared by the small-T and Tensor Core routes.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

// 2^(s-127) as FP32: a normal number for s in [1,254], the subnormal 2^-127 for s = 0.
// s = 255 is not a valid stored scale (the converter rejects it); it decodes to +inf.
__device__ __forceinline__ float mxfp8_ue8m0_to_float(unsigned s) {
    return __uint_as_float(s != 0 ? s << 23 : 0x00400000u);
}

// The same power of two broadcast into both halves of a BF16 pair (bf16 shares FP32's exponent
// field, so the value is exact: 2^-127 is the BF16 subnormal 0x0040).
__device__ __forceinline__ unsigned mxfp8_ue8m0_to_bf16x2_bits(unsigned s) {
    const unsigned bits = s != 0 ? s << 7 : 0x0040u;
    return bits | (bits << 16);
}

// Two E4M3FN codes (low byte first) widened exactly to FP32 through the FP16 pair conversion.
__device__ __forceinline__ float2 mxfp8_e4m3x2_to_float2(unsigned packed) {
    const __half2_raw raw =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed & 0xffffu), __NV_E4M3);
    return __half22float2(__half2(raw));
}

// Two E4M3FN codes widened exactly to a BF16 pair (low byte -> low half).
__device__ __forceinline__ unsigned mxfp8_e4m3x2_to_bf16x2_bits(unsigned packed) {
    const float2 v            = mxfp8_e4m3x2_to_float2(packed);
    const __nv_bfloat162 pair = __floats2bfloat162_rn(v.x, v.y);
    return *reinterpret_cast<const unsigned*>(&pair);
}

// code pair * 2^(s-127) as a BF16 pair. The product of an E4M3 value (4 significant bits) and a
// power of two is exact in BF16 whenever it is not below the BF16 normal range (|w| >= 2^-126).
__device__ __forceinline__ unsigned mxfp8_scaled_bf16x2_bits(unsigned packed, unsigned scale_bits) {
    const unsigned codes         = mxfp8_e4m3x2_to_bf16x2_bits(packed);
    const __nv_bfloat162 product = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&codes),
                                           *reinterpret_cast<const __nv_bfloat162*>(&scale_bits));
    return *reinterpret_cast<const unsigned*>(&product);
}

// ----- integer widening for the Tensor Core route ------------------------------------------------
//
// An E4M3FN code s.eeee.mmm placed as s.0000eeee.mmm0000 is the BF16 pattern of
// e4m3(code) * 2^-120 for every finite code: normal codes (e >= 1) give 2^(e-127) * 1.mmm, and the
// subnormal codes (e = 0) land on the BF16 subnormals 2^-126 * 0.mmm, both exactly 2^-120 times the
// E4M3 value. The NaN codes 0x7f / 0xff (absent from checkpoints) widen to the finite
// 480 * 2^-120 instead of NaN.

// Codes Pair*2 and Pair*2+1 of a little-endian 4-code word as a shifted BF16 pair (low code -> low
// half). prmt places each code in the low byte of its half and replicates its sign bit over the
// high byte; one shift by 4 and one mask then leave exactly the sign bit and the 7 code bits.
template <int Pair>
__device__ __forceinline__ unsigned mxfp8_e4m3x2_to_bf16x2_shifted_bits(unsigned codes) {
    constexpr unsigned kSelector = Pair == 0 ? 0x9180u : 0xb3a2u;
    unsigned spread;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(spread) : "r"(codes), "n"(kSelector));
    return (spread << 4) & 0x87f087f0u;
}

// One BF16 multiply by 2^(s-7) completes the decode when that factor is a normal BF16 number
// (s <= 134, i.e. every weight below 2^8 * 448): the product of the exact shifted code and a power
// of two is rounded once, exactly like e4m3(code) * 2^(s-127).
__device__ __forceinline__ bool mxfp8_ue8m0_direct(unsigned s) { return s <= 134u; }

__device__ __forceinline__ unsigned mxfp8_ue8m0_to_shifted_bf16x2_bits(unsigned s) {
    const unsigned bits = (s + 120u) << 7;
    return bits | (bits << 16);
}

__device__ __forceinline__ unsigned mxfp8_bf16x2_mul_bits(unsigned a, unsigned b) {
    const __nv_bfloat162 product = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a),
                                           *reinterpret_cast<const __nv_bfloat162*>(&b));
    return *reinterpret_cast<const unsigned*>(&product);
}

// Any scale: the shifted pair times 2^120 is the exact E4M3 pair, then one rounded multiply by
// 2^(s-127) as in mxfp8_scaled_bf16x2_bits.
__device__ __forceinline__ unsigned mxfp8_scale_shifted_bf16x2_bits(unsigned shifted, unsigned s) {
    constexpr unsigned kTwoTo120 = (247u << 7) | (247u << 23);
    return mxfp8_bf16x2_mul_bits(mxfp8_bf16x2_mul_bits(shifted, kTwoTo120),
                                 mxfp8_ue8m0_to_bf16x2_bits(s));
}

__device__ __forceinline__ float2 mxfp8_bf16x2_bits_to_float2(unsigned bits) {
    return make_float2(__uint_as_float(bits << 16), __uint_as_float(bits & 0xffff0000u));
}

} // namespace ninfer::ops::detail
