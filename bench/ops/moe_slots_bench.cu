// Measures moe_slots_nvfp4 (NVFP4 W4A4 MoE over expert-cache slots) at the Qwen3.8-Flash-Next
// geometry: T = 1..8 positions, U distinct resident experts per call, a slot pool of --pool
// slots (one array per plane), the remaining routed positions not resident (-1 through the
// table) ("sparse") or filled with resident experts already used by other tokens ("reuse").
// One CUDA graph holds --calls calls, each with its own slot table, resident set and slots, so
// consecutive calls read different experts (weights are loaded evict-first by the kernel).
// Reported: per-call median and minimum over --repeat graph launches, and GB/s = U * 2 764 800 B
// per call time. One JSON object per line.
#include "ninfer/ops/moe_slots.h"

#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kH = ops::kMoeSlotsHidden, kK = ops::kMoeSlotsTopK, kE = 512;
constexpr double kExpertBytes = 2764800.0;

struct Options {
    std::vector<int> tokens{1, 2, 4, 8};
    std::vector<int> resident{10, 20, 30, 40};
    std::vector<std::string> fills{"sparse", "reuse"};
    int pool = 181, calls = 24, warmup = 5, repeat = 30;
    bool anchors = false;
};

std::vector<int> int_list(const std::string& text) {
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

std::vector<std::string> str_list(const std::string& text) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find(',', start);
        out.push_back(text.substr(start, end - start));
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
            std::puts("usage: ninfer_moe_slots_bench [--tokens 1,2,4,8] [--resident 10,20,30,40] "
                      "[--fill sparse,reuse] [--anchors] [--pool 181] [--calls 24] [--warmup 5] "
                      "[--repeat 30]\n  --anchors: the F0-A synthetic points instead (reuse fill, "
                      "U = 10/17/28/42 at T = 1/2/4/8)");
            std::exit(0);
        }
        if (arg == "--anchors") {
            o.anchors = true;
            continue;
        }
        if (++i >= argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[i];
        if (arg == "--tokens") o.tokens = int_list(value);
        else if (arg == "--resident") o.resident = int_list(value);
        else if (arg == "--fill") o.fills = str_list(value);
        else if (arg == "--pool") o.pool = std::stoi(value);
        else if (arg == "--calls") o.calls = std::stoi(value);
        else if (arg == "--warmup") o.warmup = std::stoi(value);
        else if (arg == "--repeat") o.repeat = std::stoi(value);
        else throw std::invalid_argument("unknown option: " + arg);
    }
    return o;
}

struct CallData {
    std::vector<int> ids, table;
    int accesses = 0; // routed positions with a resident expert
};

// U resident experts (random ids, random distinct slots); every resident expert appears at least
// once; rows hold distinct ids.
CallData make_call(int tokens, int resident, bool reuse, int pool, std::mt19937& rng) {
    std::vector<int> ids(kE);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<int> slots(pool);
    std::iota(slots.begin(), slots.end(), 0);
    std::shuffle(slots.begin(), slots.end(), rng);
    CallData c;
    c.table.assign(kE, -1);
    for (int j = 0; j < resident; ++j) c.table[ids[j]] = slots[j];
    c.ids.assign(static_cast<std::size_t>(tokens) * kK, -1);
    std::vector<int> fill(tokens, 0);
    for (int j = 0; j < resident; ++j) {
        const int t                 = j % tokens;
        c.ids[t * kK + fill[t]++] = ids[j];
    }
    int next_missing = resident;
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> pool_ids(ids.begin(), ids.begin() + resident);
        std::shuffle(pool_ids.begin(), pool_ids.end(), rng);
        auto in_row = [&](int id) {
            for (int k = 0; k < fill[t]; ++k)
                if (c.ids[t * kK + k] == id) return true;
            return false;
        };
        if (reuse)
            for (const int id : pool_ids)
                if (fill[t] < kK && !in_row(id)) c.ids[t * kK + fill[t]++] = id;
        while (fill[t] < kK) c.ids[t * kK + fill[t]++] = ids[next_missing++ % kE];
    }
    for (const int id : c.ids) c.accesses += (id >= 0 && c.table[id] >= 0) ? 1 : 0;
    return c;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // slot pool, one array per plane: random codes, scale words 0x38 (1.0)
        const std::int64_t plane[4] = {
            ops::kMoeSlotsGateUpCodeBytes, ops::kMoeSlotsGateUpScaleBytes,
            ops::kMoeSlotsDownCodeBytes, ops::kMoeSlotsDownScaleBytes};
        std::vector<DeviceBuffer> planes;
        for (int q = 0; q < 4; ++q) {
            const std::size_t bytes = static_cast<std::size_t>(plane[q]) * o.pool;
            if (q % 2 == 0) planes.push_back(bench::make_bf16(bytes / 2, 31U + q, -3.0F, 3.0F));
            else {
                planes.emplace_back(bytes);
                planes.back().fill(0x38);
            }
        }
        ops::MoeSlotPool pool;
        pool.gate_up_codes  = {planes[0].p, plane[0]};
        pool.gate_up_scales = {planes[1].p, plane[1]};
        pool.down_codes     = {planes[2].p, plane[2]};
        pool.down_scales    = {planes[3].p, plane[3]};
        pool.slots          = o.pool;
        DeviceBuffer div13 = bench::make_f32(2 * kE, 41U, 400.0F, 600.0F);
        DeviceBuffer div2  = bench::make_f32(kE, 42U, 400.0F, 600.0F);
        DeviceBuffer x     = bench::make_bf16(
            static_cast<std::size_t>(ops::kMoeSlotsMaxTokens) * kH, 43U, -2.0F, 2.0F);
        DeviceBuffer out(static_cast<std::size_t>(ops::kMoeSlotsMaxTokens) * kH * sizeof(float));
        WorkspaceArena workspace(ops::moe_slots_nvfp4_workspace_bytes(ops::kMoeSlotsMaxTokens));
        std::vector<float> weights(static_cast<std::size_t>(ops::kMoeSlotsMaxTokens) * kK, 0.1F);
        DeviceBuffer w(weights.size() * sizeof(float));
        w.copy_from_host(weights.data(), weights.size() * sizeof(float));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::printf("{\"device\":\"%s\",\"sms\":%d,\"pool_slots\":%d,\"calls_per_graph\":%d,"
                    "\"repeat\":%d}\n",
                    prop.name, prop.multiProcessorCount, o.pool, o.calls, o.repeat);

        struct Point {
            std::string fill;
            int tokens, resident;
        };
        std::vector<Point> points;
        if (o.anchors) {
            points = {{"reuse", 1, 10}, {"reuse", 2, 17}, {"reuse", 4, 28}, {"reuse", 8, 42}};
        } else {
            for (const auto& f : o.fills)
                for (const int t : o.tokens)
                    for (const int u : o.resident)
                        if (u <= t * kK) points.push_back({f, t, u});
        }
        std::mt19937 rng(7U);
        for (const Point& p : points) {
            if (p.fill != "sparse" && p.fill != "reuse") throw std::invalid_argument("bad --fill");
            if (p.resident > o.pool) throw std::invalid_argument("U exceeds the pool");
            std::vector<DeviceBuffer> ids, tables;
            double accesses = 0;
            for (int c = 0; c < o.calls; ++c) {
                const CallData d = make_call(p.tokens, p.resident, p.fill == "reuse", o.pool, rng);
                ids.emplace_back(d.ids.size() * sizeof(int));
                ids.back().copy_from_host(d.ids.data(), d.ids.size() * sizeof(int));
                tables.emplace_back(d.table.size() * sizeof(int));
                tables.back().copy_from_host(d.table.data(), d.table.size() * sizeof(int));
                accesses += d.accesses;
            }
            bench::TimedGraph graph;
            graph.capture(stream, [&](cudaStream_t s) {
                for (int c = 0; c < o.calls; ++c) {
                    Tensor tx(x.p, DType::BF16, {kH, p.tokens});
                    Tensor ti(ids[c].p, DType::I32, {kK, p.tokens});
                    Tensor tw(w.p, DType::FP32, {kK, p.tokens});
                    Tensor to(out.p, DType::FP32, {kH, p.tokens});
                    ops::MoeSlotsLayer layer;
                    layer.slot_table            = Tensor(tables[c].p, DType::I32, {kE});
                    layer.gate_up_divisors      = Tensor(div13.p, DType::FP32, {2 * kE});
                    layer.down_divisors         = Tensor(div2.p, DType::FP32, {kE});
                    layer.gate_up_input_divisor = 600.0F;
                    layer.down_input_divisor    = 2000.0F;
                    ops::moe_slots_nvfp4(tx, ti, tw, layer, pool, ops::MoeSlotsOutput::Store, to,
                                         workspace, s);
                }
            });
            const bench::ColdTiming t = bench::measure_graph(graph, stream, o.warmup, o.repeat);
            const double median = t.median_us / o.calls, best = t.min_us / o.calls;
            std::printf("{\"fill\":\"%s\",\"T\":%d,\"U\":%d,\"resident_accesses\":%.2f,"
                        "\"us_median\":%.2f,\"us_min\":%.2f,\"GBps_median\":%.1f}\n",
                        p.fill.c_str(), p.tokens, p.resident, accesses / o.calls, median, best,
                        p.resident * kExpertBytes / median / 1e3);
            std::fflush(stdout);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_moe_slots_bench: %s\n", e.what());
        return 1;
    }
}
