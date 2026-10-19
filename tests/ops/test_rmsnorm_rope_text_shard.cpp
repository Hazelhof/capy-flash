// rmsnorm_rope_text_shard (ninfer/ops/tp2/rmsnorm_rope.h): the 27B two-device attention shard's
// query/key norm and RoPE in one launch. Two checks per case: an independent FP64 oracle of the
// documented formula, and exact equality with the three Ops it replaces (rmsnorm of q, rmsnorm of
// k, rope of the pair), which is the Op's bit-identity contract.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/tp2/rmsnorm_rope.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim    = 256;
constexpr int kRotary     = 64;
constexpr int kHalf       = kRotary / 2;
constexpr int kQueryHeads = 12;
constexpr int kKeyHeads   = 2;
constexpr float kTheta    = 1.0e7F;
constexpr float kEps      = 1.0e-6F;
// BF16 output of a normalized and rotated value, bounded by the FP64 norm of its rotation pair
// (either member can cancel), as the RoPE suite's pair profile; one BF16 rounding of the
// materialized norm and one of the rotation.
constexpr double kPairRelative = 1.4e-2;

std::vector<float> bf16_values(std::size_t count, std::uint32_t seed, float low, float high) {
    std::vector<float> values(count);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = bf16_to_f32(f32_to_bf16(distribution(generator))); }
    return values;
}

std::vector<std::uint16_t> bits_of(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

// FP64 oracle of the documented formula for one [256, heads, T] tensor.
int check_oracle(const std::string& label, const std::vector<float>& x,
                 const std::vector<float>& weight, const std::vector<int>& positions, int axes,
                 int heads, int tokens, const std::vector<std::uint16_t>& got) {
    int failures = 0;
    double worst = 0.0;
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            const std::size_t base =
                (static_cast<std::size_t>(token) * heads + static_cast<std::size_t>(head)) *
                kHeadDim;
            double sum = 0.0;
            for (int d = 0; d < kHeadDim; ++d) { sum += double(x[base + d]) * x[base + d]; }
            const double inv = 1.0 / std::sqrt(sum / kHeadDim + double(kEps));
            std::vector<double> n(kHeadDim);
            for (int d = 0; d < kHeadDim; ++d) {
                n[d] = double(x[base + d]) * inv * (1.0 + double(weight[d]));
            }
            for (int d = 0; d < kHeadDim; ++d) {
                double expected = n[d];
                double scale    = std::abs(n[d]);
                if (d < kRotary) {
                    const int pair   = d % kHalf;
                    const int axis   = axes == 3 ? pair % 3 : 0;
                    const double pos = positions[static_cast<std::size_t>(axis) * tokens + token];
                    const double angle =
                        pos * std::pow(double(kTheta), -2.0 * pair / double(kRotary));
                    const double c = std::cos(angle), s = std::sin(angle);
                    const double first = n[pair], second = n[pair + kHalf];
                    expected = d < kHalf ? first * c - second * s : second * c + first * s;
                    scale    = std::hypot(first, second);
                }
                const double observed = bf16_to_f32(got[base + d]);
                const double error    = std::abs(observed - expected);
                const double ratio    = scale > 0.0 ? error / scale : error;
                worst                 = std::max(worst, ratio);
                if (!(ratio <= kPairRelative) && failures++ == 0) {
                    std::cerr << label << ": oracle mismatch token " << token << " head " << head
                              << " d " << d << " got " << observed << " expected " << expected
                              << '\n';
                }
            }
        }
    }
    if (error_stats_enabled()) { std::cout << label << ": worst pair-relative " << worst << '\n'; }
    return failures != 0 ? 1 : 0;
}

int run_case(int tokens, int axes, int first_position, std::uint32_t seed) {
    const std::string label = "rmsnorm_rope_text_shard T=" + std::to_string(tokens) +
                              " axes=" + std::to_string(axes) +
                              " p0=" + std::to_string(first_position);
    const std::size_t q_count = static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens;
    const std::size_t k_count = static_cast<std::size_t>(kHeadDim) * kKeyHeads * tokens;
    // Activations as wide as a real projection's, including large-magnitude heads.
    std::vector<float> q        = bf16_values(q_count, seed, -6.0F, 6.0F);
    std::vector<float> k        = bf16_values(k_count, seed + 1, -40.0F, 40.0F);
    const std::vector<float> wq = bf16_values(kHeadDim, seed + 2, -0.9F, 2.0F);
    const std::vector<float> wk = bf16_values(kHeadDim, seed + 3, -0.9F, 2.0F);
    std::vector<int> positions(static_cast<std::size_t>(axes) * tokens);
    for (int axis = 0; axis < axes; ++axis) {
        for (int token = 0; token < tokens; ++token) {
            positions[static_cast<std::size_t>(axis) * tokens + token] =
                first_position + 37 * axis + (axis + 1) * token;
        }
    }

    const DeviceBuffer dq = to_device(bits_of(q)), dk = to_device(bits_of(k));
    const DeviceBuffer dwq = to_device(bits_of(wq)), dwk = to_device(bits_of(wk));
    const DeviceBuffer dpos = to_device_i32(positions);
    GuardedDeviceBuffer fused_q(q_count * 2), fused_k(k_count * 2);
    const DeviceBuffer ref_q(q_count * 2), ref_k(k_count * 2);
    const Tensor positions_t =
        axes == 1 ? Tensor(dpos.p, DType::I32, {tokens}) : Tensor(dpos.p, DType::I32, {tokens, 3});
    const Tensor q_t(dq.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    const Tensor k_t(dk.p, DType::BF16, {kHeadDim, kKeyHeads, tokens});
    const Tensor wq_t(dwq.p, DType::BF16, {kHeadDim});
    const Tensor wk_t(dwk.p, DType::BF16, {kHeadDim});
    Tensor fq(fused_q.data(), DType::BF16, {kHeadDim, kQueryHeads, tokens});
    Tensor fk(fused_k.data(), DType::BF16, {kHeadDim, kKeyHeads, tokens});
    Tensor rq(ref_q.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    Tensor rk(ref_k.p, DType::BF16, {kHeadDim, kKeyHeads, tokens});

    ops::rmsnorm_rope_text_shard(positions_t, kRotary, kTheta, wq_t, wk_t, kEps, q_t, k_t, fq, fk,
                                 nullptr);
    ops::rmsnorm(q_t, wq_t, kEps, true, rq, nullptr);
    ops::rmsnorm(k_t, wk_t, kEps, true, rk, nullptr);
    ops::rope(positions_t, kRotary, kTheta, rq, rk, nullptr);
    cuda_synchronize();

    int failures     = fused_q.verify_guards(label + " q") + fused_k.verify_guards(label + " k");
    const auto got_q = from_device<std::uint16_t>(fused_q.data(), q_count);
    const auto got_k = from_device<std::uint16_t>(fused_k.data(), k_count);
    const auto exp_q = from_device<std::uint16_t>(ref_q, q_count);
    const auto exp_k = from_device<std::uint16_t>(ref_k, k_count);
    failures += verify_exact((label + " q == rmsnorm+rope").c_str(), got_q, exp_q);
    failures += verify_exact((label + " k == rmsnorm+rope").c_str(), got_k, exp_k);
    failures += check_oracle(label + " q", q, wq, positions, axes, kQueryHeads, tokens, got_q);
    failures += check_oracle(label + " k", k, wk, positions, axes, kKeyHeads, tokens, got_k);
    // Inputs are read-only.
    failures += verify_exact((label + " q input").c_str(), from_device<std::uint16_t>(dq, q_count),
                             bits_of(q));
    return failures;
}

int expect_invalid(const std::string& what, const auto& body) {
    try {
        body();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << what << ": accepted\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "rmsnorm_rope_text_shard: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures       = 0;
    std::uint32_t seed = 0x5100U;
    for (const int tokens : {1, 2, 3, 4, 8, 16, 33, 1024}) {
        failures += run_case(tokens, 1, tokens == 1 ? 0 : 262'000, seed += 8);
        failures += run_case(tokens, 3, 131'000, seed += 8);
    }

    const DeviceBuffer a(256 * 12 * 4 * 2), b(256 * 2 * 4 * 2), c(256 * 12 * 4 * 2),
        d(256 * 2 * 4 * 2), w(512), pos(16);
    const Tensor positions(pos.p, DType::I32, {4});
    const Tensor weight(w.p, DType::BF16, {256});
    Tensor q(a.p, DType::BF16, {256, 12, 4}), k(b.p, DType::BF16, {256, 2, 4});
    Tensor qo(c.p, DType::BF16, {256, 12, 4}), ko(d.p, DType::BF16, {256, 2, 4});
    failures += expect_invalid("rotary 128", [&] {
        ops::rmsnorm_rope_text_shard(positions, 128, kTheta, weight, weight, kEps, q, k, qo, ko,
                                     nullptr);
    });
    failures += expect_invalid("in place", [&] {
        ops::rmsnorm_rope_text_shard(positions, 64, kTheta, weight, weight, kEps, q, k, q, ko,
                                     nullptr);
    });
    failures += expect_invalid("24 query heads", [&] {
        Tensor wide(c.p, DType::BF16, {256, 24, 2});
        Tensor wide_in(a.p, DType::BF16, {256, 24, 2});
        ops::rmsnorm_rope_text_shard(positions.slice(0, 0, 2), 64, kTheta, weight, weight, kEps,
                                     wide_in, k, wide, ko, nullptr);
    });

    if (failures != 0) {
        std::cerr << "rmsnorm_rope_text_shard failures=" << failures << '\n';
        return 1;
    }
    std::cout << "rmsnorm_rope_text_shard: PASS\n";
    return 0;
}
