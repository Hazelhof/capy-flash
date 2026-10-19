// MXFP8_E4M3FN_G32_UE8M0 (BlockScaleK32) x BF16 Linear against an independent FP64 oracle.
//
// The weight is generated directly in the stored format (E4M3FN code plane, 256-byte padding,
// UE8M0 scale plane, both row-major) and decoded for the oracle from the format definition:
// w = e4m3(code) * 2^(scale - 127). Every Qwen3.8-Flash-Next FP4-Local dense geometry (tp 1 and
// one tp 2 rank) is checked at decode and prefill T with BF16 and FP32 outputs.
//
//   ninfer_linear_mxfp8_a16_test            conformance (ctest)
//   ninfer_linear_mxfp8_a16_test --bench    cold-L2 timing next to Q8_G32 on the same shapes

#include "core/decode_graph.h"
#include "core/device.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;

struct Geometry {
    std::int32_t n;
    std::int32_t k;
    const char* role;
};

constexpr std::array kGeometries{
    Geometry{13312, 2560, "tp1 QSA q|k|gate|v"}, Geometry{640, 2560, "indexer q|k"},
    Geometry{2560, 6144, "tp1 attn/GDN out"},    Geometry{16384, 2560, "tp1 GDN q|k|v|z"},
    Geometry{96, 2560, "tp1 GDN a|b"},           Geometry{1280, 2560, "shared gate|up"},
    Geometry{2560, 640, "shared down"},          Geometry{6656, 2560, "tp2 q|k|gate|v"},
    Geometry{8192, 2560, "tp2 GDN q|k|v|z"},     Geometry{48, 2560, "tp2 GDN a|b"},
    Geometry{2560, 3072, "tp2 attn/GDN out"},
};

// ----- independent format oracle --------------------------------------------------------------

double decode_e4m3fn_reference(std::uint8_t code) {
    const int sign     = code >> 7;
    const int exponent = (code >> 3) & 0xf;
    const int mantissa = code & 0x7;
    if (exponent == 0xf && mantissa == 0x7) throw std::invalid_argument("E4M3FN NaN code");
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return sign ? -magnitude : magnitude;
}

double decode_ue8m0_reference(std::uint8_t scale) {
    if (scale == 0xff) throw std::invalid_argument("UE8M0 NaN scale");
    return std::ldexp(1.0, static_cast<int>(scale) - 127);
}

// One MXFP8 parent in its stored byte form. `extra_rows` precede the logical rows so a row view
// (a slice of a fused parent) starts at a non-zero row of both planes.
struct Mxfp8Host {
    std::int32_t n          = 0;
    std::int32_t k          = 0;
    std::int32_t row_offset = 0;
    std::vector<std::uint8_t> payload;
    std::size_t scale_offset = 0;

    std::uint8_t code(std::int32_t row, std::int32_t col) const {
        return payload[static_cast<std::size_t>(row_offset + row) * k + col];
    }

    std::uint8_t scale(std::int32_t row, std::int32_t group) const {
        return payload[scale_offset + static_cast<std::size_t>(row_offset + row) * (k / 32) +
                       group];
    }

    double weight(std::int32_t row, std::int32_t col) const {
        return decode_e4m3fn_reference(code(row, col)) *
               decode_ue8m0_reference(scale(row, col / 32));
    }
};

Mxfp8Host make_mxfp8(std::int32_t n, std::int32_t k, std::uint32_t seed, std::int32_t row_offset) {
    Mxfp8Host host;
    host.n                  = n;
    host.k                  = k;
    host.row_offset         = row_offset;
    const std::size_t rows  = static_cast<std::size_t>(n + row_offset);
    const std::size_t codes = rows * k;
    host.scale_offset       = (codes + 255) / 256 * 256;
    host.payload.assign(host.scale_offset + rows * (k / 32), 0);
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < codes; ++i) {
        std::uint8_t c;
        do { c = static_cast<std::uint8_t>(rng()); } while ((c & 0x7f) == 0x7f); // no NaN codes
        host.payload[i] = c;
    }
    // Realistic UE8M0 exponents (checkpoint weights ~2^-12 .. 2^-4 per unit code), with sparse
    // extremes inside the normal BF16 range of every decoded product.
    std::uniform_int_distribution<int> common(110, 124);
    std::uniform_int_distribution<int> rare(100, 140);
    for (std::size_t i = 0; i < rows * (k / 32); ++i) {
        host.payload[host.scale_offset + i] =
            static_cast<std::uint8_t>(rng() % 64 == 0 ? rare(rng) : common(rng));
    }
    return host;
}

Weight device_weight(const Mxfp8Host& host, const void* device_payload) {
    const auto* base = static_cast<const std::uint8_t*>(device_payload);
    Weight w;
    w.payload       = device_payload;
    w.payload_bytes = host.payload.size();
    w.qtype         = QType::MXFP8_E4M3FN_G32_UE8M0;
    w.layout        = QuantLayout::BlockScaleK32;
    w.ndim          = 2;
    w.n = w.shape[0] = w.padded_shape[0] = host.n;
    w.k = w.shape[1] = w.padded_shape[1] = host.k;
    w.qdata  = base + static_cast<std::size_t>(host.row_offset) * host.k;
    w.scales = base + host.scale_offset + static_cast<std::size_t>(host.row_offset) * (host.k / 32);
    w.qhigh  = nullptr;
    w.group_size  = 32;
    w.group       = 32;
    w.scale_dtype = DType::U8;
    w.scale_ne[0] = host.k / 32;
    w.scale_ne[1] = host.n;
    w.scale_nb[0] = 1;
    w.scale_nb[1] = host.k / 32;
    w.scale_nb[2] = w.scale_nb[3] = static_cast<std::int64_t>(host.k / 32) * host.n;
    return w;
}

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (std::int32_t token = 0; token < t; ++token) {
        // Per-token magnitude spread so large and small columns share one comparison.
        const float magnitude = std::ldexp(1.0F, static_cast<int>(rng() % 7) - 3);
        for (std::int32_t col = 0; col < k; ++col)
            bits[static_cast<std::size_t>(token) * k + col] =
                test::f32_to_bf16(normal(rng) * magnitude);
    }
    return bits;
}

std::vector<std::int32_t> oracle_rows(std::int32_t n) {
    if (n <= 1280) {
        std::vector<std::int32_t> all(static_cast<std::size_t>(n));
        for (std::int32_t i = 0; i < n; ++i) all[static_cast<std::size_t>(i)] = i;
        return all;
    }
    // Tile edges at both ends, the 64-row boundaries around the middle, and a spread.
    std::vector<std::int32_t> rows;
    for (std::int32_t i = 0; i < 20; ++i) rows.push_back(i);
    for (std::int32_t i = n - 20; i < n; ++i) rows.push_back(i);
    for (std::int32_t d : {-17, -9, -8, -1, 0, 1, 8, 15, 16}) rows.push_back(n / 2 / 64 * 64 + d);
    for (std::int32_t s = 0; s < 32; ++s)
        rows.push_back(
            static_cast<std::int32_t>((static_cast<std::int64_t>(n) * (2 * s + 1)) / 64));
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

// reference[row_index * t + token], FP64 over the decoded weight and the BF16 activation values.
std::vector<double> oracle(const Mxfp8Host& host, const std::vector<std::uint16_t>& x_bits,
                           std::int32_t t, const std::vector<std::int32_t>& rows) {
    const std::int32_t k = host.k;
    std::vector<double> x(static_cast<std::size_t>(k) * t);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = test::bf16_to_f32(x_bits[i]);
    std::vector<double> out(rows.size() * static_cast<std::size_t>(t));
    std::vector<double> w(static_cast<std::size_t>(k));
    for (std::size_t r = 0; r < rows.size(); ++r) {
        for (std::int32_t col = 0; col < k; ++col)
            w[static_cast<std::size_t>(col)] = host.weight(rows[r], col);
        for (std::int32_t token = 0; token < t; ++token) {
            const double* xt = x.data() + static_cast<std::size_t>(token) * k;
            double acc       = 0.0;
            for (std::int32_t col = 0; col < k; ++col)
                acc += w[static_cast<std::size_t>(col)] * xt[col];
            out[r * static_cast<std::size_t>(t) + token] = acc;
        }
    }
    return out;
}

// BF16 out: the A16 Linear criterion (one BF16 unit roundoff relative L2; gross two units of the
// largest reference). FP32 out: the dequantized weight and the BF16 activation are exact
// operands, so only FP32 accumulation error remains.
constexpr test::ReductionCriterion kBf16Criterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
constexpr test::ReductionCriterion kF32Criterion{1.0e-5, 1.0e-6, 2.0e-5};

constexpr std::uint8_t kPoison = 0xff; // BF16 0xffff / FP32 0xffffffff are NaN

int run_case(const std::string& label, const Mxfp8Host& host, const Weight& weight,
             const DeviceBuffer& x, const std::vector<double>& reference, std::int32_t t_max,
             const std::vector<std::int32_t>& rows, std::int32_t t, DType out_dtype, bool graph) {
    const std::size_t word       = out_dtype == DType::FP32 ? 4 : 2;
    const std::size_t count      = static_cast<std::size_t>(host.n) * t;
    constexpr std::size_t kGuard = 256;
    DeviceBuffer out(count * word + kGuard);
    test::cuda_check(cudaMemset(out.p, kPoison, out.bytes), "poison output");
    Tensor input(x.p, DType::BF16, {host.k, t});
    Tensor destination(out.p, out_dtype, {host.n, t});
    const std::size_t capacity = ops::linear_workspace_capacity_bytes(
        weight.qtype, host.n, host.k, ops::LinearPolicy::A16Only, t, t);
    DeviceArena workspace(256);
    int failures = 0;
    if (capacity != 0) {
        std::cerr << label << ": MXFP8 Linear must not need workspace\n";
        ++failures;
    }
    if (graph) {
        DeviceContext context;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        test::cuda_check(cudaDeviceSynchronize(), "settle before capture");
        definition.capture(context.stream, [&] {
            ops::linear(input, weight, destination, ops::LinearPolicy::A16Only, workspace,
                        context.stream);
        });
        executable.instantiate(definition);
        executable.launch(context.stream);
        test::cuda_check(cudaStreamSynchronize(context.stream), "graph replay");
    } else {
        ops::linear(input, weight, destination, ops::LinearPolicy::A16Only, workspace, nullptr);
        test::cuda_check(cudaDeviceSynchronize(), "linear");
    }

    std::vector<std::uint8_t> bytes(out.bytes);
    out.copy_to_host(bytes.data(), bytes.size());
    for (std::size_t i = count * word; i < bytes.size(); ++i) {
        if (bytes[i] != kPoison) {
            std::cerr << label << ": wrote past the [N,T] output\n";
            ++failures;
            break;
        }
    }
    const auto value = [&](std::size_t index) -> double {
        if (out_dtype == DType::FP32) {
            float f;
            std::memcpy(&f, bytes.data() + index * 4, 4);
            return f;
        }
        std::uint16_t h;
        std::memcpy(&h, bytes.data() + index * 2, 2);
        return test::bf16_to_f32(h);
    };
    std::size_t non_finite = 0;
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(value(i))) ++non_finite;
    if (non_finite) {
        std::cerr << label << ": " << non_finite << " unwritten or non-finite outputs\n";
        ++failures;
    }
    std::vector<double> got, ref;
    got.reserve(rows.size() * t);
    ref.reserve(rows.size() * t);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        for (std::int32_t token = 0; token < t; ++token) {
            got.push_back(value(static_cast<std::size_t>(token) * host.n + rows[r]));
            ref.push_back(reference[r * static_cast<std::size_t>(t_max) + token]);
        }
    }
    failures += test::verify_reduction(label, got, ref,
                                       out_dtype == DType::FP32 ? kF32Criterion : kBf16Criterion);
    return failures;
}

int conformance() {
    int failures                 = 0;
    constexpr std::int32_t kMaxT = 512;
    const std::vector<std::int32_t> token_counts{1, 2,  3,  4,  5,  6,   7,  8,
                                                 9, 16, 33, 64, 65, 100, 512};
    std::uint32_t seed = 9001;
    for (const auto& g : kGeometries) {
        // Odd shapes also run as a row view one row into a larger parent (a fused-group slice).
        for (const std::int32_t row_offset : {0, 1}) {
            if (row_offset == 1 && g.n > 2560) continue;
            const Mxfp8Host host = make_mxfp8(g.n, g.k, seed++, row_offset);
            DeviceBuffer payload(host.payload.size());
            payload.copy_from_host(host.payload.data(), payload.bytes);
            const Weight weight = device_weight(host, payload.p);
            const auto x_bits   = make_activation(g.k, kMaxT, seed++);
            DeviceBuffer x(x_bits.size() * 2);
            x.copy_from_host(x_bits.data(), x.bytes);
            const auto rows      = oracle_rows(g.n);
            const auto reference = oracle(host, x_bits, kMaxT, rows);
            int shape_failures   = 0;
            for (const std::int32_t t : token_counts) {
                for (const DType dtype : {DType::BF16, DType::FP32}) {
                    const bool graph = (t == 1 || t == 64) && dtype == DType::BF16;
                    const std::string label =
                        std::string("MXFP8_A16 [") + std::to_string(g.n) + "," +
                        std::to_string(g.k) + "]+" + std::to_string(row_offset) +
                        " T=" + std::to_string(t) + (dtype == DType::FP32 ? " f32" : " bf16") +
                        (graph ? " graph" : "");
                    try {
                        shape_failures += run_case(label, host, weight, x, reference, kMaxT, rows,
                                                   t, dtype, graph);
                    } catch (const std::exception& error) {
                        std::cerr << label << ": unexpected exception: " << error.what() << '\n';
                        ++shape_failures;
                    }
                }
            }
            std::vector<std::uint8_t> after(host.payload.size());
            payload.copy_to_host(after.data(), after.size());
            if (after != host.payload) {
                std::cerr << g.role << ": linear modified its weight\n";
                ++shape_failures;
            }
            std::cout << (shape_failures ? "FAIL " : "ok   ") << "[" << g.n << "," << g.k << "]+"
                      << row_offset << " " << g.role << '\n';
            failures += shape_failures;
        }
    }
    return failures;
}

// Admission: other shapes, malformed metadata and an FP32 output of another format are rejected.
int admission() {
    int failures         = 0;
    const Mxfp8Host host = make_mxfp8(640, 2560, 7, 0);
    DeviceBuffer payload(host.payload.size());
    payload.copy_from_host(host.payload.data(), payload.bytes);
    DeviceBuffer x(2560 * 2), out(640 * 4);
    Tensor input(x.p, DType::BF16, {2560, 1});
    Tensor destination(out.p, DType::BF16, {640, 1});
    const auto expect_throw = [&](const char* what, auto&& call) {
        try {
            call();
            std::cerr << "admission: accepted " << what << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    Weight bad      = device_weight(host, payload.p);
    bad.scale_dtype = DType::FP16;
    expect_throw("FP16 scale metadata", [&] { ops::linear(input, bad, destination, nullptr); });
    bad        = device_weight(host, payload.p);
    bad.layout = QuantLayout::RowScale;
    expect_throw("RowScale layout", [&] { ops::linear(input, bad, destination, nullptr); });
    expect_throw("unregistered shape", [&] {
        (void)ops::linear_workspace_capacity_bytes(QType::MXFP8_E4M3FN_G32_UE8M0, 640, 2048,
                                                   ops::LinearPolicy::A16Only, 1, 1);
    });
    test::cuda_check(cudaDeviceSynchronize(), "admission");
    return failures;
}

// ----- informative microbenchmark --------------------------------------------------------------

struct Timing {
    double microseconds;
};

template <class Launch>
Timing time_cold(Launch&& launch, int iterations) {
    // Rewriting 128 MiB between calls evicts the 48 MiB L2, so each call streams its weight.
    DeviceBuffer flush(128ULL << 20);
    cudaEvent_t start, stop;
    test::cuda_check(cudaEventCreate(&start), "event");
    test::cuda_check(cudaEventCreate(&stop), "event");
    launch(); // warm-up (module load, attributes)
    std::vector<float> samples;
    for (int i = 0; i < iterations; ++i) {
        test::cuda_check(cudaMemsetAsync(flush.p, i & 0xff, flush.bytes), "flush");
        test::cuda_check(cudaEventRecord(start), "record");
        launch();
        test::cuda_check(cudaEventRecord(stop), "record");
        test::cuda_check(cudaEventSynchronize(stop), "sync");
        float ms = 0;
        test::cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        samples.push_back(ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2] * 1000.0};
}

int bench() {
    std::cout << "shape            T   | MXFP8 us   GB/s  TFLOPS | Q8_G32 us   GB/s  TFLOPS\n";
    for (const auto& g : kGeometries) {
        const Mxfp8Host host = make_mxfp8(g.n, g.k, 77, 0);
        DeviceBuffer mx(host.payload.size());
        mx.copy_from_host(host.payload.data(), mx.bytes);
        const Weight mx_weight = device_weight(host, mx.p);
        const auto q8_host     = test::linear::make_q8_g32_fp16_weight(g.n, g.k, 78);
        DeviceBuffer q8(q8_host.payload.size());
        q8.copy_from_host(q8_host.payload.data(), q8.bytes);
        const Weight q8_weight       = q8_host.device_weight(q8.p);
        const std::uint64_t mx_bytes = static_cast<std::uint64_t>(g.n) * g.k * 33 / 32;
        const std::uint64_t q8_bytes = q8_host.payload.size();
        for (const std::int32_t t : {1, 8, 64, 512}) {
            DeviceBuffer x(static_cast<std::size_t>(g.k) * t * 2),
                out(static_cast<std::size_t>(g.n) * t * 2);
            test::cuda_check(cudaMemset(x.p, 0x3c, x.bytes), "x");
            Tensor input(x.p, DType::BF16, {g.k, t});
            Tensor destination(out.p, DType::BF16, {g.n, t});
            const int iterations = t == 1 ? 50 : 10;
            const auto mx_time =
                time_cold([&] { ops::linear(input, mx_weight, destination, nullptr); }, iterations);
            const auto q8_time =
                time_cold([&] { ops::linear(input, q8_weight, destination, nullptr); }, iterations);
            const double flops = 2.0 * g.n * g.k * t;
            const auto row     = [&](Timing time, std::uint64_t bytes) {
                std::cout << std::setw(9) << std::fixed << std::setprecision(1) << time.microseconds
                          << std::setw(7) << std::setprecision(0)
                          << bytes / (time.microseconds * 1e3) << std::setw(8)
                          << std::setprecision(2) << flops / (time.microseconds * 1e6);
            };
            std::cout << "[" << std::setw(5) << g.n << "," << std::setw(4) << g.k << "] "
                      << std::setw(4) << t << " |";
            row(mx_time, mx_bytes);
            std::cout << " |";
            row(q8_time, q8_bytes);
            std::cout << '\n';
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        if (argc > 1 && std::string_view(argv[1]) == "--bench") return bench();
        const int failures = conformance() + admission();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " MXFP8_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "MXFP8_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
