// T43: the Qwen4Exp draft shortlist proposal head (execution/qwen4_draft_head.h).
//
// CPU: the shortlist split (longest ID prefix, cap, rounding to 8 rows with the lowest missing
// IDs, errors) and the file reader. GPU (skips without a device): a random Q6 RowSplit head
// [20000, 2560] (random codes, FP16 scales in [2^-7, 2^-6)), a shortlist = ID prefix [0, 8000)
// + 1003 random IDs in [8000, 20000) (rounded to 9008 rows), B = 1 and B = 2:
//   * the gathered rows are byte-identical to the head rows they copy;
//   * every shortlist logit equals, bit for bit, the full-head logit of its token computed by
//     the same T = 1 route over the whole [20000, 2560] head (a row's GEMV does not depend on the
//     other rows): the prefix view and the gather address the right rows;
//   * the proposed token is the argmax over the shortlist of the full-head logits (lowest ID on
//     ties) and is in the shortlist;
//   * the log-probability is log softmax over the shortlist rows at the proposed row: FP64 host
//     oracle from the same BF16 logits, |error| <= 1e-4 (FP32 accumulation over 9008 terms of
//     exp(l - max) <= 1: relative error ~1e-6, so 1e-4 nats has two orders of margin).
#include "models/qwen3_5/execution/qwen4_draft_head.h"

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models::qwen3_5::execution;

int failures = 0;
void check(bool ok, const std::string& what) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << '\n';
    if (!ok) { ++failures; }
}

template <class F> bool throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) { return true; }
    return false;
}

void split_cases() {
    {
        std::vector<std::int32_t> ids;
        for (int i = 0; i < 64; ++i) { ids.push_back(63 - i); }
        ids.push_back(100);
        for (int i = 0; i < 7; ++i) { ids.push_back(200 + i); }
        const auto s = split_draft_shortlist(ids, 1000, 1000);
        check(s.prefix == 64 && s.extras.size() == 8 && s.extras.front() == 100 &&
                  s.extras.back() == 206,
              "split: prefix [0,64) + 8 ascending extras");
        const auto c = split_draft_shortlist(ids, 1000, 40);
        check(c.prefix == 40 && c.extras.size() == 32 && c.extras.front() == 40,
              "split: prefix capped at the limit, the rest are extras");
        const auto r = split_draft_shortlist(ids, 1000, 37);
        check(r.prefix == 32 && r.extras.front() == 32, "split: prefix rounded down to 8 rows");
    }
    {
        // 13 IDs: completed to 16 with the lowest IDs it lacks (0 is present: 1, 2, 3).
        std::vector<std::int32_t> ids{0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 500};
        const auto s = split_draft_shortlist(ids, 1000, 1000);
        const std::size_t rows = static_cast<std::size_t>(s.prefix) + s.extras.size();
        std::set<std::int32_t> all(s.extras.begin(), s.extras.end());
        for (std::int32_t i = 0; i < s.prefix; ++i) { all.insert(i); }
        check(rows == 16 && all.count(1) && all.count(2) && all.count(3) && all.count(500) &&
                  s.prefix == 0,
              "split: rounded up to 16 rows with IDs 1,2,3 (prefix 0..3 < 8 -> 0)");
    }
    check(throws([] { (void)split_draft_shortlist(std::vector<std::int32_t>{1, 2, 2}, 10, 10); }),
          "split: duplicate ID rejected");
    check(throws([] { (void)split_draft_shortlist(std::vector<std::int32_t>{1, 10}, 10, 10); }),
          "split: ID >= vocabulary rejected");
    check(throws([] { (void)split_draft_shortlist(std::vector<std::int32_t>{}, 10, 10); }),
          "split: empty shortlist rejected");
    {
        const std::string path = "/tmp/ninfer_qwen4_draft_head_test.txt";
        {
            std::ofstream f(path);
            f << "# a comment 1 2 3\n5 6\n7,8\n\n9 # trailing\n10\n";
        }
        const auto all = read_draft_shortlist(path.c_str(), 0);
        const auto two = read_draft_shortlist(path.c_str(), 2);
        check(all == std::vector<std::int32_t>{5, 6, 7, 8, 9, 10} &&
                  two == std::vector<std::int32_t>{5, 6},
              "reader: comments, separators, row cap");
        std::remove(path.c_str());
    }
}

std::uint16_t bf16(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
double from_bf16(std::uint16_t b) {
    const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float v = 0;
    std::memcpy(&v, &bits, 4);
    return v;
}

void gpu_cases() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "skip: no CUDA device for the GPU cases\n";
        return;
    }
    constexpr std::int32_t N = 20000, K = 2560;
    const std::array<std::size_t, 3> row{K / 64 * 32, K / 64 * 16, K / 64 * 2};
    std::mt19937 rng(43);
    std::vector<std::uint8_t> codes(N * row[0]), high(N * row[1]);
    std::vector<std::uint16_t> scales(N * row[2] / 2);
    for (auto& c : codes) { c = static_cast<std::uint8_t>(rng()); }
    for (auto& h : high) { h = static_cast<std::uint8_t>(rng()); }
    for (auto& s : scales) { s = static_cast<std::uint16_t>(0x2000 + rng() % 0x400); }
    std::array<void*, 3> plane{};
    CUDA_CHECK(cudaMalloc(&plane[0], codes.size()));
    CUDA_CHECK(cudaMalloc(&plane[1], high.size()));
    CUDA_CHECK(cudaMalloc(&plane[2], scales.size() * 2));
    CUDA_CHECK(cudaMemcpy(plane[0], codes.data(), codes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(plane[1], high.data(), high.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(plane[2], scales.data(), scales.size() * 2, cudaMemcpyHostToDevice));
    LinearParameters head;
    Weight& w        = head.weight;
    w.qtype          = QType::Q6_G64_FP16;
    w.layout         = QuantLayout::RowSplit;
    w.group_size     = 64;
    w.group          = 64;
    w.n              = N;
    w.k              = K;
    w.ndim           = 2;
    w.shape[0]       = N;
    w.shape[1]       = K;
    w.padded_shape[0] = N;
    w.padded_shape[1] = K;
    w.qdata          = plane[0];
    w.qhigh          = plane[1];
    w.scales         = plane[2];
    w.payload        = plane[0];
    check(draft_head_supported(w), "Q6 RowSplit head accepted");

    // Shortlist: [0, 8000) + 1003 random IDs above.
    std::vector<std::int32_t> ids;
    for (std::int32_t i = 0; i < 8000; ++i) { ids.push_back(i); }
    std::set<std::int32_t> picked;
    while (picked.size() < 1003) { picked.insert(8000 + static_cast<std::int32_t>(rng() % 12000)); }
    ids.insert(ids.end(), picked.begin(), picked.end());
    std::shuffle(ids.begin(), ids.end(), rng);
    const auto split = split_draft_shortlist(ids, N, N);
    const std::int32_t P = split.prefix, M = static_cast<std::int32_t>(split.extras.size());
    check(P == 8000 && P + M == 9008, "GPU shortlist: prefix 8000, 9008 rows");

    std::array<std::byte*, 3> extra{};
    for (std::size_t p = 0; p < 3; ++p) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&extra[p]), row[p] * M));
    }
    std::vector<std::int32_t> map(static_cast<std::size_t>(P + M));
    for (std::int32_t i = 0; i < P; ++i) { map[static_cast<std::size_t>(i)] = i; }
    std::copy(split.extras.begin(), split.extras.end(), map.begin() + P);
    std::int32_t* map_d = nullptr;
    CUDA_CHECK(cudaMalloc(&map_d, map.size() * 4));
    CUDA_CHECK(cudaMemcpy(map_d, map.data(), map.size() * 4, cudaMemcpyHostToDevice));
    gather_q6_rows(w, split.extras, extra, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        bool same = true;
        const std::array<const std::uint8_t*, 3> src{codes.data(), high.data(),
                                                     reinterpret_cast<const std::uint8_t*>(scales.data())};
        for (std::size_t p = 0; p < 3 && same; ++p) {
            std::vector<std::uint8_t> got(row[p] * M);
            CUDA_CHECK(cudaMemcpy(got.data(), extra[p], got.size(), cudaMemcpyDeviceToHost));
            for (std::int32_t i = 0; i < M && same; ++i) {
                same = std::memcmp(got.data() + row[p] * i,
                                   src[p] + row[p] * split.extras[static_cast<std::size_t>(i)],
                                   row[p]) == 0;
            }
        }
        check(same, "gathered rows are byte-identical to the head rows");
    }
    Qwen4DraftHead draft;
    draft.prefix_rows = P;
    draft.extra_rows  = M;
    draft.prefix      = q6_row_view(head, 0, P);
    draft.extra       = q6_rows_weight(head, extra, M);
    draft.token_ids   = map_d;

    WorkspaceArena work(1 << 20);
    for (const std::int32_t B : {1, 2}) {
        std::vector<std::uint16_t> xh(static_cast<std::size_t>(K) * B);
        std::uniform_real_distribution<float> u(-1.0F, 1.0F);
        for (auto& v : xh) { v = bf16(u(rng)); }
        void *x_d = nullptr, *full_d = nullptr, *logits_d = nullptr, *tok_d = nullptr,
             *lp_d = nullptr;
        CUDA_CHECK(cudaMalloc(&x_d, xh.size() * 2));
        CUDA_CHECK(cudaMalloc(&full_d, static_cast<std::size_t>(N) * B * 2));
        CUDA_CHECK(cudaMalloc(&logits_d, static_cast<std::size_t>(N) * B * 2));
        CUDA_CHECK(cudaMalloc(&tok_d, B * 4));
        CUDA_CHECK(cudaMalloc(&lp_d, B * 4));
        CUDA_CHECK(cudaMemcpy(x_d, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
        // Reference: the whole head through the same T = 1 route, one column at a time.
        for (std::int32_t b = 0; b < B; ++b) {
            Tensor xb(static_cast<std::uint16_t*>(x_d) + static_cast<std::size_t>(b) * K,
                      DType::BF16, {K, 1});
            Tensor ob(static_cast<std::uint16_t*>(full_d) + static_cast<std::size_t>(b) * N,
                      DType::BF16, {N, 1});
            ops::linear(xb, w, ob, head.policy, work, nullptr);
        }
        Tensor x(x_d, DType::BF16, {K, B});
        Tensor logits(logits_d, DType::BF16, {N, B});
        Tensor tokens(tok_d, DType::I32, {B});
        Tensor logprobs(lp_d, DType::FP32, {B});
        draft_head_propose(draft, x, logits, tokens, &logprobs, work, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint16_t> full(static_cast<std::size_t>(N) * B),
            got(static_cast<std::size_t>(P + M) * B);
        std::vector<std::int32_t> tok(B);
        std::vector<float> lp(B);
        CUDA_CHECK(cudaMemcpy(full.data(), full_d, full.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got.data(), logits_d, got.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tok.data(), tok_d, B * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lp.data(), lp_d, B * 4, cudaMemcpyDeviceToHost));
        bool bits = true, argmax = true, member = true, logprob = true;
        double worst = 0;
        for (std::int32_t b = 0; b < B; ++b) {
            const std::uint16_t* f = full.data() + static_cast<std::size_t>(b) * N;
            const std::uint16_t* g = got.data() + static_cast<std::size_t>(b) * (P + M);
            std::int32_t best = -1;
            double best_v = -1e300, mx = -1e300;
            for (std::int32_t i = 0; i < P + M; ++i) {
                const std::int32_t id = map[static_cast<std::size_t>(i)];
                bits = bits && g[i] == f[id];
                const double v = from_bf16(f[id]);
                if (v > best_v || (v == best_v && id < best)) { best_v = v; best = id; }
                mx = std::max(mx, from_bf16(g[i]));
            }
            argmax = argmax && tok[static_cast<std::size_t>(b)] == best;
            member = member && std::find(map.begin(), map.end(), tok[static_cast<std::size_t>(b)]) != map.end();
            double sum = 0;
            for (std::int32_t i = 0; i < P + M; ++i) { sum += std::exp(from_bf16(g[i]) - mx); }
            const double ref = best_v - mx - std::log(sum);
            worst = std::max(worst, std::fabs(ref - lp[static_cast<std::size_t>(b)]));
            logprob = logprob && std::fabs(ref - lp[static_cast<std::size_t>(b)]) <= 1e-4;
        }
        const std::string tag = " (B = " + std::to_string(B) + ")";
        check(bits, "shortlist logits = full-head logits of their tokens, bit for bit" + tag);
        check(argmax && member, "proposal = shortlist argmax of the full-head logits" + tag);
        char buf[96];
        std::snprintf(buf, sizeof buf, "log-probability vs FP64 oracle, max |error| %.2e <= 1e-4", worst);
        check(logprob, buf + tag);
        for (void* p : {x_d, full_d, logits_d, tok_d, lp_d}) { CUDA_CHECK(cudaFree(p)); }
    }
    for (auto* p : extra) { CUDA_CHECK(cudaFree(p)); }
    for (auto* p : plane) { CUDA_CHECK(cudaFree(p)); }
    CUDA_CHECK(cudaFree(map_d));
}

} // namespace

int main() {
    try {
        split_cases();
        gpu_cases();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4 draft head (" << failures
              << " failures)\n";
    return failures == 0 ? 0 : 1;
}
