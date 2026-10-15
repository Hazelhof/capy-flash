// Measures one Qwen4Exp PLE block at the Qwen3.8-Flash-Next geometry as a CUDA Graph replay:
// the Q8 key/value projection [12800,2560] of the n-gram embedding, ple_gate, the conv
// grouped_offset_rmsnorm, ple_dilated_conv and ple_conv_advance. `--part ops` times the PLE Ops
// without the projection.
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/ple.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kStreams = 4, kHidden = 2560, kWide = kStreams * kHidden, kEmbed = 2560;
constexpr int kTaps = 4, kDilation = 3, kHistory = (kTaps - 1) * kDilation;

struct Options {
    int warmup = 20, repeat = 200;
    std::vector<int> tokens{1, 2, 4, 8, 64};
    std::string part = "block", cache = "cold";
};

std::vector<int> list(const std::string& text) {
    std::vector<int> out;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find(',', start);
        out.push_back(std::stoi(text.substr(start, end - start)));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::puts("usage: ninfer_ple_bench [--part block|ops] [--tokens T,...] "
                      "[--cache cold|warm] [--warmup N] [--repeat N]");
            std::exit(0);
        }
        if (++i >= argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[i];
        if (arg == "--part") o.part = value;
        else if (arg == "--tokens") o.tokens = list(value);
        else if (arg == "--cache") o.cache = value;
        else if (arg == "--warmup") o.warmup = std::stoi(value);
        else if (arg == "--repeat") o.repeat = std::stoi(value);
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if ((o.part != "block" && o.part != "ops") || (o.cache != "cold" && o.cache != "warm"))
        throw std::invalid_argument("invalid part or cache");
    return o;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        int devices     = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) return 77;
        DeviceContext device;
        bench::L2FlushBuffer flush(std::size_t{256} << 20);
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU=%s CUDART=%d part=%s cache=%s\n", prop.name, CUDART_VERSION,
                    o.part.c_str(), o.cache.c_str());
        auto kv = bench::make_row_split_weight(QType::Q8_G32_FP16, kWide + kHidden, kEmbed,
                                               kEmbed, 0x81U, 0.05F);
        DeviceBuffer qn = bench::make_bf16(kWide, 0x82U, -0.1F, 0.1F);
        DeviceBuffer kn = bench::make_bf16(kWide, 0x83U, -0.1F, 0.1F);
        DeviceBuffer cn = bench::make_bf16(kWide, 0x84U, -0.1F, 0.1F);
        DeviceBuffer conv = bench::make_bf16(std::size_t(kWide) * kTaps, 0x85U, -0.3F, 0.3F);
        DeviceBuffer state = bench::make_bf16(std::size_t(kWide) * kHistory, 0x86U, -1.0F, 1.0F);
        DeviceBuffer slot(4);
        CUDA_CHECK(cudaMemset(slot.p, 0, 4));
        Tensor t_qn(qn.p, DType::BF16, {kWide}), t_kn(kn.p, DType::BF16, {kWide}),
            t_cn(cn.p, DType::BF16, {kWide}), t_conv(conv.p, DType::BF16, {kWide, kTaps}),
            t_state(state.p, DType::BF16, {kHistory * kWide, 1}), t_slot(slot.p, DType::I32, {1});
        for (const int t : o.tokens) {
            DeviceBuffer embedding = bench::make_bf16(std::size_t(kEmbed) * t, 0x87U, -0.02F, 0.02F);
            DeviceBuffer residual  = bench::make_bf16(std::size_t(kWide) * t, 0x88U, -0.75F, 0.75F);
            DeviceBuffer key_value(std::size_t(kWide + kHidden) * t * 2);
            DeviceBuffer gated(std::size_t(kWide) * t * 2), normed(std::size_t(kWide) * t * 2);
            Tensor t_e(embedding.p, DType::BF16, {kEmbed, t});
            Tensor t_r(residual.p, DType::BF16, {kWide, t});
            Tensor t_kv(key_value.p, DType::BF16, {kWide + kHidden, t});
            Tensor t_g(gated.p, DType::BF16, {kWide, t});
            Tensor t_n(normed.p, DType::BF16, {kWide, t});
            CUDA_CHECK(cudaMemset(key_value.p, 0, key_value.bytes));
            const auto launch = [&](cudaStream_t stream) {
                if (o.part == "block") ops::linear(t_e, kv.weight, t_kv, stream);
                ops::ple_gate(t_r, t_kv, t_qn, t_kn, kStreams, 1e-6F, t_g, stream);
                ops::grouped_offset_rmsnorm(t_g, t_cn, kStreams, 1e-6F, t_n, stream);
                Tensor n3 = t_n.view({kWide, t, 1}), g3 = t_g.view({kWide, t, 1}),
                       r3 = t_r.view({kWide, t, 1});
                ops::ple_dilated_conv(n3, g3, t_conv, kDilation, t_state, t_slot, r3, stream);
                ops::ple_conv_advance(n3, nullptr, kHistory, t_state, t_slot, t_slot, stream);
            };
            launch(device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            bench::TimedGraph graph;
            graph.capture(device.stream, launch);
            const bench::ColdTiming time =
                o.cache == "cold"
                    ? bench::measure_cold_graph(graph, flush, device.stream, o.warmup, o.repeat)
                    : bench::measure_graph(graph, device.stream, o.warmup, o.repeat);
            std::printf("ple %s T=%d cache=%s nodes=%zu median=%.2f min=%.2f p95=%.2f us "
                        "(kv weight %.1f MB)\n",
                        o.part.c_str(), t, o.cache.c_str(), graph.nodes(), time.median_us,
                        time.min_us, time.p95_us, kv.model_weight_bytes() / 1e6);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ple_bench: %s\n", error.what());
        return 1;
    }
}
