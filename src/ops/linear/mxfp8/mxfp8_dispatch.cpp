#include "ops/linear/mxfp8/mxfp8_dispatch.h"

#include "ops/linear/mxfp8/mxfp8_operands.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct Mxfp8Problem {
    std::int32_t n;
    std::int32_t k;
};

// Qwen3.8-Flash-Next FP4-Local dense projections (MXFP8 in the NVIDIA checkpoint). One rank at
// tp 1: QSA q|k|gate|v, QSA indexer q|k, attention/GDN output, GDN q|k|v|z, GDN a|b, shared
// expert gate|up and down. One rank's shard at tp 2: q|k|gate|v, q|k|v|z, a|b and the
// row-parallel outputs (the indexer and the shared expert stay whole on one rank).
constexpr std::array kProblems{
    Mxfp8Problem{13312, 2560}, Mxfp8Problem{640, 2560},  Mxfp8Problem{2560, 6144},
    Mxfp8Problem{16384, 2560}, Mxfp8Problem{96, 2560},   Mxfp8Problem{1280, 2560},
    Mxfp8Problem{2560, 640},   Mxfp8Problem{6656, 2560}, Mxfp8Problem{8192, 2560},
    Mxfp8Problem{48, 2560},    Mxfp8Problem{2560, 3072},
};

void require_problem(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("mxfp8 linear: unsupported policy");
    for (const auto& problem : kProblems) {
        if (problem.n == n && problem.k == k) return;
    }
    throw std::invalid_argument("mxfp8 linear: unsupported shape");
}

bool aligned(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0;
}

void validate_mxfp8_weight(const Weight& w) {
    if (w.layout != QuantLayout::BlockScaleK32 || w.scale_dtype != DType::U8 ||
        w.group_size != 32 || w.qhigh != nullptr) {
        throw std::invalid_argument(
            "mxfp8 linear: weight must be BlockScaleK32 with UE8M0 (U8) scales per 32 columns");
    }
    if (w.k % 32 != 0 || w.padded_shape[1] != w.k) {
        throw std::invalid_argument("mxfp8 linear: K must be a multiple of 32 without padding");
    }
    // Code rows are K bytes (K % 32 == 0), so a 16-byte aligned plane keeps every row aligned.
    // Scale rows are K/32 bytes; the GEMM stages them as aligned 4-byte words (every registered
    // K has K/32 % 4 == 0, and a row view of a 256-byte aligned plane stays 4-byte aligned).
    if (!aligned(w.qdata, 16) || !aligned(w.scales, 4) || (w.k / 32) % 4 != 0) {
        throw std::invalid_argument(
            "mxfp8 linear: code plane must be 16-byte and scale plane 4-byte aligned");
    }
}

} // namespace

std::size_t mxfp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("mxfp8 linear workspace: invalid token interval");
    require_problem(n, k, policy);
    return 0;
}

void mxfp8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                    cudaStream_t stream) {
    require_problem(w.n, w.k, policy);
    validate_mxfp8_weight(w);
    const auto operands = mxfp8_linear_operands(x, w);
    if (operands.tokens <= 0) throw std::invalid_argument("mxfp8 linear: T must be positive");
    if (mxfp8_uses_gemv(operands.rows, operands.tokens)) {
        launch_mxfp8_a16_gemv(operands, out, stream);
    } else {
        launch_mxfp8_a16_mma(operands, out, stream);
    }
}

} // namespace ninfer::ops::detail
