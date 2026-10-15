// moe_predict_ahead against an FP64 oracle (include/ninfer/ops/moe_predict.h).
//
// Oracle: the contract's formula in FP64 from the represented BF16 inputs, with the two BF16
// roundings the contract specifies (the normalized streams n and the stream mean u). The Op
// reduces in FP32, so (a) n or u may land on the other side of a BF16 rounding boundary for a few
// elements and (b) each logit carries the FP32 accumulation error of 2560 products. Both move a
// window score by far less than kScoreTol = 2e-4 per token (measured worst ~1e-6; a one-ulp flip
// of u[h] moves a logit by |router[e,h]| * 2^-8 * |u[h]| < 1e-3 and the score by p * that). So:
//   * every published score equals the oracle score of its expert within kScoreTol * T;
//   * candidates are distinct, eligible (not resident, oracle score >= min_score - tol), in
//     non-increasing oracle order within tol, and each has an oracle score >= the oracle's
//     count-th eligible score - tol (selection exact up to near-ties);
//   * past the eligible experts the outputs are -1 / 0; the flag is 1 (mapped memory).
#include "core/device.h"
#include "ninfer/ops/moe_predict.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int H = ops::kMoePredictHidden;
constexpr int S = ops::kMoePredictStreams;
constexpr int E = ops::kMoePredictExperts;
constexpr double kScoreTol = 2e-4;
constexpr float kEps       = 1e-6F;

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

float bf16_round(double v) { return bf16_to_f32(f32_to_bf16(static_cast<float>(v))); }

struct Mapped {
    std::byte* p = nullptr;
    explicit Mapped(std::size_t bytes) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&p), bytes, cudaHostAllocMapped));
        std::memset(p, 0, bytes);
    }
    ~Mapped() { (void)cudaFreeHost(p); }
};

int run_case(const std::string& label, int T, int count, int residents, float min_score,
             std::uint32_t seed, bool graph, int repeats = 1) {
    std::vector<float> x(static_cast<std::size_t>(S) * H * T), norm(static_cast<std::size_t>(S) * H),
        router(static_cast<std::size_t>(E + 1) * H);
    fill_uniform(x, seed, -1.0F, 1.0F);
    // Streams of very different scales (the RMS normalization must remove them).
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < S; ++s) {
            const float scale = 0.05F * static_cast<float>(1 + 7 * s + t);
            for (int h = 0; h < H; ++h) x[(static_cast<std::size_t>(t) * S + s) * H + h] *= scale;
        }
    }
    fill_uniform(norm, seed + 1, -0.5F, 0.5F);
    fill_uniform(router, seed + 2, -0.05F, 0.05F);
    std::mt19937 rng(seed + 3);
    std::uniform_real_distribution<float> row_scale(0.2F, 4.0F);
    for (int e = 0; e <= E; ++e) {
        const float sc = row_scale(rng);
        for (int h = 0; h < H; ++h) router[static_cast<std::size_t>(e) * H + h] *= sc;
    }
    round_to_bf16(x);
    round_to_bf16(norm);
    round_to_bf16(router);
    std::vector<std::int32_t> resident(E, -1);
    {
        std::vector<int> ids(E);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        for (int i = 0; i < residents; ++i) resident[static_cast<std::size_t>(ids[i])] = i;
    }

    // Oracle.
    std::vector<double> score(E, 0.0);
    for (int t = 0; t < T; ++t) {
        const float* xc = x.data() + static_cast<std::size_t>(t) * S * H;
        double inv[S];
        for (int s = 0; s < S; ++s) {
            double ss = 0;
            for (int h = 0; h < H; ++h) ss += double(xc[s * H + h]) * xc[s * H + h];
            inv[s] = 1.0 / std::sqrt(ss / H + kEps);
        }
        std::vector<double> u(H);
        for (int h = 0; h < H; ++h) {
            double mean = 0;
            for (int s = 0; s < S; ++s) {
                mean += bf16_round(double(xc[s * H + h]) * inv[s] * (1.0 + norm[s * H + h]));
            }
            u[h] = bf16_round(mean / S);
        }
        std::vector<double> l(E);
        double m = -1e300;
        for (int e = 0; e < E; ++e) {
            double sum = 0;
            for (int h = 0; h < H; ++h) sum += double(router[static_cast<std::size_t>(e) * H + h]) * u[h];
            l[e] = sum;
            m    = std::max(m, sum);
        }
        double z = 0;
        for (int e = 0; e < E; ++e) z += std::exp(l[e] - m);
        for (int e = 0; e < E; ++e) score[e] += std::exp(l[e] - m) / z;
    }
    const double tol = kScoreTol * T;
    std::vector<int> eligible;
    for (int e = 0; e < E; ++e) {
        if (resident[e] < 0 && score[e] >= min_score) eligible.push_back(e);
    }
    std::stable_sort(eligible.begin(), eligible.end(), [&](int a, int b) { return score[a] > score[b]; });

    // Device.
    const auto xb = encode_bf16(x), nb = encode_bf16(norm), rb = encode_bf16(router);
    GuardedDeviceBuffer dx(xb.size() * 2), dn(nb.size() * 2), dr(rb.size() * 2), dres(E * 4),
        dwork(ops::moe_predict_workspace_bytes());
    dx.copy_from_host(xb.data(), dx.bytes());
    dn.copy_from_host(nb.data(), dn.bytes());
    dr.copy_from_host(rb.data(), dr.bytes());
    dres.copy_from_host(resident.data(), dres.bytes());
    dwork.fill(0);
    Mapped out(1024);
    auto* cands  = reinterpret_cast<std::int32_t*>(out.p);
    auto* scores = reinterpret_cast<float*>(out.p + 256);
    auto* flag   = reinterpret_cast<std::uint32_t*>(out.p + 512);
    std::fill(cands, cands + 32, 12345);
    Tensor tx(dx.data(), DType::BF16, {S * H, T});
    Tensor tn(dn.data(), DType::BF16, {S * H});
    Tensor tr(dr.data(), DType::BF16, {H, E + 1});
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto launch = [&] {
        ops::moe_predict_ahead(tx, tn, tr, kEps, static_cast<const std::int32_t*>(dres.data()),
                               min_score, count, cands, scores, flag, dwork.data(), stream);
    };
    int failures = 0;
    for (int rep = 0; rep < repeats; ++rep) {
        *reinterpret_cast<volatile std::uint32_t*>(flag) = 0;
        if (graph) {
            cudaGraph_t captured;
            cudaGraphExec_t exec;
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            launch();
            CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
            CUDA_CHECK(cudaGraphInstantiate(&exec, captured, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaGraphExecDestroy(exec));
            CUDA_CHECK(cudaGraphDestroy(captured));
        } else {
            launch();
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        if (*reinterpret_cast<volatile std::uint32_t*>(flag) != 1) {
            std::cerr << label << ": flag not set\n";
            ++failures;
        }
        // The completion counter is reset for the next call.
        const auto counter = from_device<std::uint32_t>(dwork.data(), 1);
        if (counter[0] != 0) {
            std::cerr << label << ": workspace counter not reset\n";
            ++failures;
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    const int expect_valid = std::min<int>(count, static_cast<int>(eligible.size()));
    const double kth = expect_valid > 0 ? score[eligible[static_cast<std::size_t>(expect_valid - 1)]] : 0.0;
    std::vector<bool> seen(E, false);
    double worst_score = 0;
    int near_ties = 0;
    for (int k = 0; k < count; ++k) {
        const int e = cands[k];
        if (k >= expect_valid) {
            // An expert at the min_score edge may be judged either way: accept a valid extra one
            // only if its score is within tol of the gate.
            if (e == -1 && scores[k] == 0.0F) continue;
            if (e >= 0 && e < E && resident[e] < 0 && std::fabs(score[e] - min_score) <= tol) continue;
            std::cerr << label << ": slot " << k << " should be empty, holds " << e << '\n';
            ++failures;
            continue;
        }
        if (e < 0 || e >= E || seen[e]) {
            std::cerr << label << ": candidate " << k << " invalid or repeated: " << e << '\n';
            ++failures;
            continue;
        }
        seen[e] = true;
        if (resident[e] >= 0) {
            std::cerr << label << ": candidate " << e << " is resident\n";
            ++failures;
        }
        if (score[e] < min_score - tol || score[e] < kth - tol) {
            std::cerr << label << ": candidate " << e << " score " << score[e] << " below the cut "
                      << kth << '\n';
            ++failures;
        }
        if (k > 0 && cands[k - 1] >= 0 && score[e] > score[cands[k - 1]] + tol) {
            std::cerr << label << ": order violated at " << k << '\n';
            ++failures;
        }
        if (e != eligible[static_cast<std::size_t>(k)]) ++near_ties;
        worst_score = std::max(worst_score, std::fabs(double(scores[k]) - score[e]));
    }
    if (worst_score > tol) {
        std::cerr << label << ": score error " << worst_score << " > " << tol << '\n';
        ++failures;
    }
    failures += dx.verify_guards(label + " x") + dn.verify_guards(label + " norm") +
                dr.verify_guards(label + " router") + dres.verify_guards(label + " resident") +
                dwork.verify_guards(label + " workspace");
    std::cout << "STATS " << label << " T=" << T << " eligible=" << eligible.size()
              << " valid=" << expect_valid << " score_err=" << worst_score
              << " near_tie_positions=" << near_ties << '\n';
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        for (const int T : {1, 2, 4, 8}) {
            failures += run_case("predict T=" + std::to_string(T), T, 16, 77, 0.0F,
                                 500U + static_cast<std::uint32_t>(T), false);
        }
        failures += run_case("predict count 32, 300 resident", 4, 32, 300, 0.0F, 41U, false);
        failures += run_case("predict score gate", 4, 16, 77, 0.05F, 42U, false);
        failures += run_case("predict gate removes all", 1, 8, 0, 2.0F, 43U, false);
        failures += run_case("predict all but 3 resident", 2, 16, 509, 0.0F, 44U, false);
        failures += run_case("predict graph, repeated", 4, 16, 77, 0.0F, 45U, true, 3);
    } catch (const std::exception& error) {
        std::cerr << "moe_predict test error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "moe_predict: " << failures << " failures\n";
        return 1;
    }
    std::cout << "OK moe_predict correctness\n";
    return 0;
}
