// T103 check (docs/capy-flash/development-log.md): the generic BF16 linear (ops/linear/bf16/
// bf16_generic.cu) reached through ops::linear() on Qwen3.8-Flash-Next's BF16 shapes and on odd
// shapes, against an FP64 host reference over sampled rows; and the generic route against the
// registered 27B kernels on a registered shape. Prints one line per case; exit 1 on a failure.
// Usage: t103_bf16_generic
#include "ninfer/ops/linear.h"
#include "ops/linear/bf16/bf16_shapes.h"

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::uint16_t to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

float from_bf16(std::uint16_t h) {
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { check(cudaMalloc(&p, std::max<std::size_t>(bytes, 16)), "malloc"); }
    ~Device() { cudaFree(p); }
};

Weight bf16_weight(const void* data, int n, int k) {
    Weight w;
    w.payload       = data;
    w.payload_bytes = static_cast<std::uint64_t>(n) * k * 2;
    w.qtype         = QType::BF16;
    w.ndim          = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.layout        = QuantLayout::Contiguous;
    w.qdata         = data;
    return w;
}

bool run_case(int n, int k, int tokens, std::mt19937& rng, bool versus_registered) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> hw(static_cast<std::size_t>(n) * k), hx(static_cast<std::size_t>(k) * tokens);
    const float ws = 1.0F / std::sqrt(static_cast<float>(k));
    for (auto& v : hw) v = to_bf16(normal(rng) * ws);
    for (auto& v : hx) v = to_bf16(normal(rng) * 2.0F);
    Device dw(hw.size() * 2), dx(hx.size() * 2), dout(static_cast<std::size_t>(n) * tokens * 2),
        dref(static_cast<std::size_t>(n) * tokens * 2);
    check(cudaMemcpy(dw.p, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice), "copy w");
    check(cudaMemcpy(dx.p, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice), "copy x");
    const Weight w = bf16_weight(dw.p, n, k);
    const Tensor x(dx.p, DType::BF16, {k, tokens});
    Tensor out(dout.p, DType::BF16, {n, tokens});
    std::vector<std::uint16_t> got(static_cast<std::size_t>(n) * tokens);
    if (versus_registered) {
        // Registered schedule (ops::linear) against the generic route on the same operands.
        Tensor ref(dref.p, DType::BF16, {n, tokens});
        ops::linear(x, w, ref, nullptr);
        ops::detail::launch_bf16_generic(x, w, out, nullptr);
        check(cudaDeviceSynchronize(), "sync");
        std::vector<std::uint16_t> want(got.size());
        check(cudaMemcpy(got.data(), dout.p, got.size() * 2, cudaMemcpyDeviceToHost), "copy out");
        check(cudaMemcpy(want.data(), dref.p, want.size() * 2, cudaMemcpyDeviceToHost), "copy ref");
        double num = 0, den = 0, maxd = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double d = from_bf16(got[i]) - from_bf16(want[i]);
            num += d * d;
            den += static_cast<double>(from_bf16(want[i])) * from_bf16(want[i]);
            maxd = std::max(maxd, std::fabs(d));
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        const bool ok    = rel < 4e-3;
        std::printf("%s generic-vs-registered n=%d k=%d T=%d relL2 %.2e max|d| %.3e\n",
                    ok ? "PASS" : "FAIL", n, k, tokens, rel, maxd);
        return ok;
    }
    ops::linear(x, w, out, nullptr);
    check(cudaDeviceSynchronize(), "sync");
    check(cudaMemcpy(got.data(), dout.p, got.size() * 2, cudaMemcpyDeviceToHost), "copy out");
    // FP64 reference over sampled rows (all rows when n <= 512).
    std::vector<int> rows;
    if (n <= 512) {
        for (int r = 0; r < n; ++r) rows.push_back(r);
    } else {
        std::uniform_int_distribution<int> pick(0, n - 1);
        for (int i = 0; i < 384; ++i) rows.push_back(pick(rng));
        rows.push_back(0);
        rows.push_back(n - 1);
    }
    std::vector<double> xf(hx.size());
    for (std::size_t i = 0; i < hx.size(); ++i) xf[i] = from_bf16(hx[i]);
    double num = 0, den = 0, maxd = 0, maxr = 0;
    for (const int r : rows) {
        const std::uint16_t* wr = hw.data() + static_cast<std::size_t>(r) * k;
        for (int t = 0; t < tokens; ++t) {
            const double* xc = xf.data() + static_cast<std::size_t>(t) * k;
            double s = 0;
            for (int c = 0; c < k; ++c) s += static_cast<double>(from_bf16(wr[c])) * xc[c];
            const double g = from_bf16(got[static_cast<std::size_t>(t) * n + r]);
            num += (g - s) * (g - s);
            den += s * s;
            maxd = std::max(maxd, std::fabs(g - s));
            maxr = std::max(maxr, std::fabs(s));
        }
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    // One BF16 rounding of the output is ~2^-9 relative on average (relL2 ~2.3e-3 at most).
    const bool ok = rel < 4e-3 && maxd <= maxr * 8e-3 + 1e-6;
    std::printf("%s n=%d k=%d T=%d rows=%zu relL2 %.2e max|d|/max|ref| %.2e\n", ok ? "PASS" : "FAIL", n, k,
                tokens, rows.size(), rel, maxd / std::max(maxr, 1e-30));
    return ok;
}

} // namespace

int main() {
    try {
        std::mt19937 rng(103);
        struct Shape {
            int n, k;
        };
        // Flash-Next BF16 projections (tp 1 and the tp 2 shards), then odd shapes (scalar route,
        // partial tiles).
        const Shape shapes[] = {{640, 2560},  {324, 10240}, {320, 10240}, {10240, 320},
                                {16384, 2560}, {8192, 2560}, {2560, 6144}, {2560, 3072},
                                {100, 72},    {33, 37},     {4100, 200}};
        const int token_counts[] = {1, 2, 3, 4, 7, 8, 9, 16, 17, 63, 64, 65, 300, 1024};
        int failures = 0, cases = 0;
        for (const auto& s : shapes) {
            for (const int t : token_counts) {
                if (static_cast<double>(s.n) * s.k * t > 4.5e10) continue;
                failures += run_case(s.n, s.k, t, rng, false) ? 0 : 1;
                ++cases;
            }
        }
        for (const int t : {1, 4, 17, 100, 640}) {
            failures += run_case(256, 5120, t, rng, true) ? 0 : 1;
            failures += run_case(5120, 3072, t, rng, true) ? 0 : 1;
            cases += 2;
        }
        std::printf("t103_bf16_generic: %d cases, %d failures\n", cases, failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "t103_bf16_generic: %s\n", e.what());
        return 2;
    }
}
