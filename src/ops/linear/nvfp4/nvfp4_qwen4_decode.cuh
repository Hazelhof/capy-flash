#pragma once
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

#include <cstdlib>

namespace ninfer::ops::detail::nvfp4_qwen4 {

// Qwen4Exp GDN projections (NVFP4-GDN recipe) at decode widths T <= 8 (T49). The shapes inherited
// the 27B sliced-K schedule (8 K-warps, 16 rows, 2 stages): 536 GB/s for [16384, 2560] and
// 412 GB/s for [2560, 6144] at T = 4 (160 CTAs: two waves on 70 SMs). Measurement option
// NINFER_T49_NVFP4=<v> selects a candidate for every shape here; unset = the shape's selection.
using PZ = Nvfp4ActivationStage;
template <int KWarps, int MinBlocks, int Stages, int RowTiles>
using Sliced = Nvfp4A16SlicedKMmaSchedule<KWarps, 8, MinBlocks, Cache::ca, Cache::cg,
                                          PZ::PaddedZero, Stages, RowTiles>;

inline int variant() {
    static const int v = [] {
        const char* e = std::getenv("NINFER_T49_NVFP4");
        return e != nullptr && *e != 0 ? std::atoi(e) : -1;
    }();
    return v;
}

template <class Geometry>
Nvfp4Launch candidate(int v) {
    constexpr int K = Geometry::kInputRows;
    using Base      = Nvfp4SlicedInstance<8, 8, 2>;
    switch (v) {
    case 1:
        return nvfp4_linear_a16_sliced_k<Geometry, Sliced<8, 2, 2, 2>>; // 32 rows per CTA
    case 2:
        return nvfp4_linear_a16_sliced_k<Geometry, Sliced<4, 3, 2, 1>>; // BlockK 256
    case 3:
        return nvfp4_linear_a16_sliced_k<Geometry, Sliced<8, 3, 2, 1>>; // 3 CTAs per SM
    case 4:
        return nvfp4_linear_a16_sliced_k<Geometry, Sliced<4, 2, 2, 2>>; // BlockK 256, 32 rows
    case 5:
        return nvfp4_linear_a16_sliced_k<Geometry, Sliced<8, 2, 1, 1>>; // one stage
    case 6:
        if constexpr (K % 1024 == 0) {
            return nvfp4_linear_a16_sliced_k<Geometry, Sliced<16, 1, 2, 1>>;
        } else {
            return nvfp4_linear_a16_sliced_k<Geometry, Base>;
        }
    case 7:
        return nvfp4_linear_a16_sliced_k<Geometry, Sliced<8, 4, 2, 1>>; // 4 CTAs per SM
    default:
        return nvfp4_linear_a16_sliced_k<Geometry, Base>;
    }
}

template <class Geometry>
Nvfp4Launch select_t8(int chosen) {
    return candidate<Geometry>(variant() >= 0 ? variant() : chosen);
}

} // namespace ninfer::ops::detail::nvfp4_qwen4
