// Qwen4Exp PLE Ops against FP64 oracles at the Qwen3.8-Flash-Next geometry: 4 streams of 2560,
// a kernel-4 causal convolution with dilation 3 (nine history columns per slot), and the complete
// PLE block composition with the Q8 key/value parent [12800,2560] of the ninfer-ext recipe.
//
// Element oracles port ninfer-ext tests/ops/test_ple.cpp; the block oracle ports the complete
// FP64 transform of ninfer6000 tests/ops/test_flash_next_ple.cpp (Lars Karlslund, Apache-2.0)
// without its BF16 intermediate roundings (they are private staging here). Criteria are this
// suite's own.
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/ple.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStreams  = 4;
constexpr std::int32_t kHidden   = 2560;
constexpr std::int32_t kChannels = kStreams * kHidden;
constexpr std::int32_t kEmbed    = 2560; // 16 heads x 160
constexpr std::int32_t kTaps     = 4;
constexpr std::int32_t kDilation = 3;
constexpr std::int32_t kHistory  = (kTaps - 1) * kDilation;
constexpr std::int32_t kSlots    = 3;
constexpr float kEps             = 1.0e-6F;

// One BF16 output rounding (2^-9 relative) with a 2x margin for FP32 exp/rsqrt.
constexpr PointwiseCriterion kBf16Output{1.0e-6, 1.0 / 128.0};
// PLE block residual: the output is R + u + silu(conv) rounded once to BF16 (relative 2^-9 of a
// value dominated by R); the private BF16 staging of key/value (linear output), gated u and the
// normalized c adds ~2^-9 relative on the smaller u and conv terms. 4 x 10^-3 relative L2 is ~2x
// that budget; the gross cap is two BF16 ulps of the largest value.
constexpr ReductionCriterion kBlock{/*relative_l2*/ 4.0e-3, /*gross_absolute*/ 1.0e-4,
                                    /*gross_relative_to_max_reference*/ 1.6e-2};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double silu(double x) { return x * sigmoid(x); }

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
    return out;
}

std::vector<float> random_bf16(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

// FP64 gate oracle for one column: residual/key rows of S*H, value of H.
void gate_reference(const double* residual, const double* key, const double* value,
                    const std::vector<float>& query_norm, const std::vector<float>& key_norm,
                    double* gated) {
    for (std::int32_t s = 0; s < kStreams; ++s) {
        const std::int32_t base = s * kHidden;
        double qq = 0.0, kk = 0.0;
        for (std::int32_t h = 0; h < kHidden; ++h) {
            qq += residual[base + h] * residual[base + h];
            kk += key[base + h] * key[base + h];
        }
        const double iq = 1.0 / std::sqrt(qq / kHidden + double(kEps));
        const double ik = 1.0 / std::sqrt(kk / kHidden + double(kEps));
        double dot      = 0.0;
        for (std::int32_t h = 0; h < kHidden; ++h) {
            dot += residual[base + h] * iq * (1.0 + query_norm[base + h]) * key[base + h] * ik *
                   (1.0 + key_norm[base + h]);
        }
        const double g    = dot / std::sqrt(double(kHidden));
        const double sign = g > 0.0 ? 1.0 : (g < 0.0 ? -1.0 : 0.0);
        const double gate = sigmoid(sign * std::sqrt(std::max(std::abs(g), 1.0e-6)));
        for (std::int32_t h = 0; h < kHidden; ++h) gated[base + h] = gate * value[h];
    }
}

int gate_case(std::int32_t tokens, std::uint32_t seed) {
    const std::size_t wide    = static_cast<std::size_t>(kChannels) * tokens;
    const std::size_t kv_rows = kChannels + kHidden;
    auto residual             = random_bf16(wide, seed, -2.0F, 2.0F);
    auto key_value            = random_bf16(kv_rows * tokens, seed + 1U, -3.0F, 3.0F);
    const auto query_norm     = random_bf16(kChannels, seed + 2U, -0.3F, 0.3F);
    const auto key_norm       = random_bf16(kChannels, seed + 3U, -0.3F, 0.3F);
    // Column 0 stream 1: a zero key gives an exactly zero dot (sign(0) and the 1e-6 floor).
    for (std::int32_t h = 0; h < kHidden; ++h) { key_value[kHidden + h] = 0.0F; }
    // Column 0 stream 2: a tiny residual exercises eps against a vanishing RMS.
    for (std::int32_t h = 0; h < kHidden; ++h) { residual[2 * kHidden + h] *= 1.0e-4F; }
    round_to_bf16(residual);
    std::vector<double> reference(wide);
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::vector<double> r(kChannels), k(kChannels), v(kHidden);
        for (std::int32_t i = 0; i < kChannels; ++i) {
            r[i] = residual[static_cast<std::size_t>(t) * kChannels + i];
            k[i] = key_value[static_cast<std::size_t>(t) * kv_rows + i];
        }
        for (std::int32_t h = 0; h < kHidden; ++h)
            v[h] = key_value[static_cast<std::size_t>(t) * kv_rows + kChannels + h];
        gate_reference(r.data(), k.data(), v.data(), query_norm, key_norm,
                       reference.data() + static_cast<std::size_t>(t) * kChannels);
    }
    DeviceBuffer dr  = to_device(bits(residual));
    DeviceBuffer dkv = to_device(bits(key_value));
    DeviceBuffer dq  = to_device(bits(query_norm));
    DeviceBuffer dk  = to_device(bits(key_norm));
    GuardedDeviceBuffer out(wide * 2);
    Tensor tr(dr.p, DType::BF16, {kChannels, tokens});
    Tensor tkv(dkv.p, DType::BF16, {static_cast<std::int32_t>(kv_rows), tokens});
    Tensor tq(dq.p, DType::BF16, {kChannels});
    Tensor tk(dk.p, DType::BF16, {kChannels});
    Tensor to(out.data(), DType::BF16, {kChannels, tokens});
    ops::ple_gate(tr, tkv, tq, tk, kStreams, kEps, to, nullptr);
    cuda_synchronize();
    const std::string label = "ple_gate T=" + std::to_string(tokens);
    return verify_pointwise(label, from_device_bf16(out.data(), wide), reference, kBf16Output) +
           out.verify_guards(label);
}

// Column m of lane b in [history ; normed]: history rows are oldest first.
double history_value(const std::vector<float>& states, const std::vector<float>& normed,
                     std::int32_t slot, std::int32_t lane, std::int32_t width, std::int32_t m,
                     std::int32_t c) {
    if (m >= 0) { return normed[(static_cast<std::size_t>(lane) * width + m) * kChannels + c]; }
    const std::int32_t row = kHistory + m;
    return states[(static_cast<std::size_t>(slot) * kHistory + row) * kChannels + c];
}

// `second_valid` >= 0 overrides lane 1's committed count (0: a cancelled speculative row, whose
// destination receives its source history unchanged).
int conv_case(std::int32_t width, std::int32_t lanes, std::uint32_t seed,
              std::int32_t second_valid = -1) {
    const std::size_t columns = static_cast<std::size_t>(width) * lanes;
    const auto normed         = random_bf16(columns * kChannels, seed, -3.0F, 3.0F);
    const auto gated          = random_bf16(columns * kChannels, seed + 1U, -2.0F, 2.0F);
    const auto residual       = random_bf16(columns * kChannels, seed + 2U, -4.0F, 4.0F);
    const auto weight =
        random_bf16(static_cast<std::size_t>(kTaps) * kChannels, seed + 3U, -0.5F, 0.5F);
    const auto states = random_bf16(static_cast<std::size_t>(kSlots) * kHistory * kChannels,
                                    seed + 4U, -3.0F, 3.0F);
    const std::vector<int> sources      = {2, 0};
    const std::vector<int> destinations = {1, 2};
    const std::vector<int> valid        = {width,
                                    second_valid >= 0 ? second_valid : (width > 1 ? width - 1 : 1)};

    std::vector<double> reference(columns * kChannels);
    std::vector<float> advanced = states;
    for (std::int32_t lane = 0; lane < lanes; ++lane) {
        const std::int32_t slot = sources[lane];
        for (std::int32_t w = 0; w < width; ++w) {
            for (std::int32_t c = 0; c < kChannels; ++c) {
                double sum = 0.0;
                for (std::int32_t j = 0; j < kTaps; ++j) {
                    const std::int32_t m = w - (kTaps - 1 - j) * kDilation;
                    sum += double(weight[static_cast<std::size_t>(j) * kChannels + c]) *
                           history_value(states, normed, slot, lane, width, m, c);
                }
                const std::size_t i = (static_cast<std::size_t>(lane) * width + w) * kChannels + c;
                reference[i]        = double(residual[i]) + double(gated[i]) + silu(sum);
            }
        }
        const std::int32_t count = valid[lane];
        for (std::int32_t r = 0; r < kHistory; ++r) {
            for (std::int32_t c = 0; c < kChannels; ++c) {
                advanced[(static_cast<std::size_t>(destinations[lane]) * kHistory + r) * kChannels +
                         c] = static_cast<float>(history_value(states, normed, slot, lane, width,
                                                               count + r - kHistory, c));
            }
        }
    }

    DeviceBuffer dn   = to_device(bits(normed));
    DeviceBuffer dg   = to_device(bits(gated));
    DeviceBuffer dr   = to_device(bits(residual));
    DeviceBuffer dw   = to_device(bits(weight));
    DeviceBuffer ds   = to_device(bits(states));
    DeviceBuffer dsrc = to_device_i32(std::vector<int>(sources.begin(), sources.begin() + lanes));
    DeviceBuffer ddst =
        to_device_i32(std::vector<int>(destinations.begin(), destinations.begin() + lanes));
    DeviceBuffer dvalid = to_device_i32(std::vector<int>(valid.begin(), valid.begin() + lanes));
    Tensor tn(dn.p, DType::BF16, {kChannels, width, lanes});
    Tensor tg(dg.p, DType::BF16, {kChannels, width, lanes});
    Tensor tr(dr.p, DType::BF16, {kChannels, width, lanes});
    Tensor tw(dw.p, DType::BF16, {kChannels, kTaps});
    Tensor ts(ds.p, DType::BF16, {kHistory * kChannels, kSlots});
    Tensor tsrc(dsrc.p, DType::I32, {lanes});
    Tensor tdst(ddst.p, DType::I32, {lanes});
    Tensor tvalid(dvalid.p, DType::I32, {lanes});
    ops::ple_dilated_conv(tn, tg, tw, kDilation, ts, tsrc, tr, nullptr);
    cuda_synchronize();
    const std::string label =
        "ple_dilated_conv W=" + std::to_string(width) + " B=" + std::to_string(lanes);
    int failures =
        verify_pointwise(label, from_device_bf16(dr, reference.size()), reference, kBf16Output);
    failures += verify_exact((label + " state immutable").c_str(),
                             from_device<std::uint16_t>(ds, states.size()), bits(states));

    ops::ple_conv_advance(tn, &tvalid, kHistory, ts, tsrc, tdst, nullptr);
    cuda_synchronize();
    failures += verify_exact(
        ("ple_conv_advance W=" + std::to_string(width) + " B=" + std::to_string(lanes)).c_str(),
        from_device<std::uint16_t>(ds, states.size()), bits(advanced));
    return failures;
}

int in_place_advance_case(std::uint32_t seed) {
    // Decode updates a slot in place: source equals destination for width one.
    const auto normed = random_bf16(kChannels, seed, -3.0F, 3.0F);
    const auto states = random_bf16(static_cast<std::size_t>(kSlots) * kHistory * kChannels,
                                    seed + 1U, -3.0F, 3.0F);
    std::vector<float> expected = states;
    const std::size_t base      = static_cast<std::size_t>(1) * kHistory * kChannels;
    for (std::int32_t r = 0; r + 1 < kHistory; ++r) {
        for (std::int32_t c = 0; c < kChannels; ++c) {
            expected[base + static_cast<std::size_t>(r) * kChannels + c] =
                states[base + static_cast<std::size_t>(r + 1) * kChannels + c];
        }
    }
    for (std::int32_t c = 0; c < kChannels; ++c) {
        expected[base + static_cast<std::size_t>(kHistory - 1) * kChannels + c] = normed[c];
    }
    DeviceBuffer dn   = to_device(bits(normed));
    DeviceBuffer ds   = to_device(bits(states));
    DeviceBuffer slot = to_device_i32({1});
    Tensor tn(dn.p, DType::BF16, {kChannels, 1, 1});
    Tensor ts(ds.p, DType::BF16, {kHistory * kChannels, kSlots});
    Tensor tslot(slot.p, DType::I32, {1});
    ops::ple_conv_advance(tn, nullptr, kHistory, ts, tslot, tslot, nullptr);
    cuda_synchronize();
    return verify_exact("ple_conv_advance in place", from_device<std::uint16_t>(ds, states.size()),
                        bits(expected));
}

double e4m3fn(std::uint8_t code) {
    const int sign = code >> 7, exponent = (code >> 3) & 0xF, mantissa = code & 7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return sign ? -magnitude : magnitude;
}

// The complete PLE block of one decode window (W = T columns, B = 1 lane, state slot 0):
// Q8 key/value projection of the decoded n-gram embedding, gate, conv norm, dilated convolution
// added to the residual, history advance. Optionally captured in a CUDA Graph.
int block_case(std::int32_t tokens, bool graph, std::uint32_t seed) {
    const std::size_t wide     = static_cast<std::size_t>(kChannels) * tokens;
    const std::int32_t kv_rows = kChannels + kHidden;
    // Embedding: FP8 E4M3 codes times a per-row BF16 scale, decoded to BF16 as the host gather
    // uploads them (16 rows of 160 per column).
    std::vector<float> embedding(static_cast<std::size_t>(kEmbed) * tokens);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t row = 0; row < embedding.size() / 160; ++row) {
        state = state * 1664525U + 1013904223U;
        const float scale =
            bf16_to_f32(f32_to_bf16(0.002F + 0.004F * float(state >> 8) / 16777216.0F));
        for (std::size_t i = 0; i < 160; ++i) {
            state             = state * 1664525U + 1013904223U;
            std::uint8_t code = static_cast<std::uint8_t>(state >> 24);
            if ((code & 0x7F) == 0x7F) code &= 0xFE; // no NaN code
            embedding[row * 160 + i] =
                bf16_to_f32(f32_to_bf16(static_cast<float>(e4m3fn(code) * scale)));
        }
    }
    const auto residual   = random_bf16(wide, seed + 1U, -0.75F, 0.75F);
    const auto query_norm = random_bf16(kChannels, seed + 2U, -0.1F, 0.1F);
    const auto key_norm   = random_bf16(kChannels, seed + 3U, -0.1F, 0.1F);
    const auto conv_norm  = random_bf16(kChannels, seed + 4U, -0.1F, 0.1F);
    const auto conv =
        random_bf16(static_cast<std::size_t>(kTaps) * kChannels, seed + 5U, -0.3F, 0.3F);
    const auto history =
        random_bf16(static_cast<std::size_t>(kHistory) * kChannels, seed + 6U, -1.0F, 1.0F);
    std::vector<float> kv_source(static_cast<std::size_t>(kv_rows) * kEmbed);
    fill_uniform(kv_source, seed + 7U, -0.05F, 0.05F);
    const auto kv = quantized_weight::pack_q8_g32_row_split(kv_source, kv_rows, kEmbed);

    // FP64 oracle.
    std::vector<double> expected(wide), expected_state(history.begin(), history.end());
    std::vector<double> window(static_cast<std::size_t>(kHistory + tokens) * kChannels);
    for (std::size_t i = 0; i < history.size(); ++i) window[i] = history[i];
    std::vector<std::vector<double>> gated_columns(tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::vector<double> key_value(kv_rows), r(kChannels), u(kChannels);
        for (std::int32_t row = 0; row < kv_rows; ++row) {
            const float* w = kv.dequant.data() + static_cast<std::size_t>(row) * kEmbed;
            double sum     = 0.0;
            for (std::int32_t k = 0; k < kEmbed; ++k)
                sum += double(w[k]) * embedding[static_cast<std::size_t>(t) * kEmbed + k];
            key_value[row] = sum;
        }
        for (std::int32_t i = 0; i < kChannels; ++i)
            r[i] = residual[static_cast<std::size_t>(t) * kChannels + i];
        gate_reference(r.data(), key_value.data(), key_value.data() + kChannels, query_norm,
                       key_norm, u.data());
        double* c = window.data() + static_cast<std::size_t>(kHistory + t) * kChannels;
        for (std::int32_t s = 0; s < kStreams; ++s) {
            double sum = 0.0;
            for (std::int32_t h = 0; h < kHidden; ++h)
                sum += u[s * kHidden + h] * u[s * kHidden + h];
            const double inv = 1.0 / std::sqrt(sum / kHidden + double(kEps));
            for (std::int32_t h = 0; h < kHidden; ++h)
                c[s * kHidden + h] = u[s * kHidden + h] * inv * (1.0 + conv_norm[s * kHidden + h]);
        }
        gated_columns[t] = u;
    }
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t ch = 0; ch < kChannels; ++ch) {
            double sum = 0.0;
            for (std::int32_t j = 0; j < kTaps; ++j) {
                const std::int32_t m = kHistory + t - (kTaps - 1 - j) * kDilation;
                sum += double(conv[static_cast<std::size_t>(j) * kChannels + ch]) *
                       window[static_cast<std::size_t>(m) * kChannels + ch];
            }
            const std::size_t i = static_cast<std::size_t>(t) * kChannels + ch;
            expected[i]         = double(residual[i]) + gated_columns[t][ch] + silu(sum);
        }
    }
    for (std::int32_t r = 0; r < kHistory; ++r)
        for (std::int32_t ch = 0; ch < kChannels; ++ch)
            expected_state[static_cast<std::size_t>(r) * kChannels + ch] =
                window[static_cast<std::size_t>(tokens + r) * kChannels + ch];

    DeviceBuffer d_embedding = to_device(bits(embedding));
    DeviceBuffer d_residual  = to_device(bits(residual));
    DeviceBuffer d_qn        = to_device(bits(query_norm));
    DeviceBuffer d_kn        = to_device(bits(key_norm));
    DeviceBuffer d_cn        = to_device(bits(conv_norm));
    DeviceBuffer d_conv      = to_device(bits(conv));
    DeviceBuffer d_state     = to_device(bits(history));
    DeviceBuffer d_kv        = to_device(kv.payload);
    DeviceBuffer d_slot      = to_device_i32({0});
    DeviceBuffer d_key_value(static_cast<std::size_t>(kv_rows) * tokens * 2);
    DeviceBuffer d_gated(wide * 2), d_normed(wide * 2);
    const Weight w_kv = kv.device_weight(d_kv.p);
    Tensor t_embedding(d_embedding.p, DType::BF16, {kEmbed, tokens});
    Tensor t_residual(d_residual.p, DType::BF16, {kChannels, tokens});
    Tensor t_key_value(d_key_value.p, DType::BF16, {kv_rows, tokens});
    Tensor t_gated(d_gated.p, DType::BF16, {kChannels, tokens});
    Tensor t_normed(d_normed.p, DType::BF16, {kChannels, tokens});
    Tensor t_qn(d_qn.p, DType::BF16, {kChannels});
    Tensor t_kn(d_kn.p, DType::BF16, {kChannels});
    Tensor t_cn(d_cn.p, DType::BF16, {kChannels});
    Tensor t_conv(d_conv.p, DType::BF16, {kChannels, kTaps});
    Tensor t_state(d_state.p, DType::BF16, {kHistory * kChannels, 1});
    Tensor t_slot(d_slot.p, DType::I32, {1});
    DeviceContext device;
    const auto launch = [&] {
        ops::linear(t_embedding, w_kv, t_key_value, device.stream);
        ops::ple_gate(t_residual, t_key_value, t_qn, t_kn, kStreams, kEps, t_gated, device.stream);
        ops::grouped_offset_rmsnorm(t_gated, t_cn, kStreams, kEps, t_normed, device.stream);
        Tensor normed3   = t_normed.view({kChannels, tokens, 1});
        Tensor gated3    = t_gated.view({kChannels, tokens, 1});
        Tensor residual3 = t_residual.view({kChannels, tokens, 1});
        ops::ple_dilated_conv(normed3, gated3, t_conv, kDilation, t_state, t_slot, residual3,
                              device.stream);
        ops::ple_conv_advance(normed3, nullptr, kHistory, t_state, t_slot, t_slot, device.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    if (graph) {
        definition.capture(device.stream, launch);
        executable.instantiate(definition);
        executable.launch(device.stream);
    } else {
        launch();
    }
    cuda_synchronize(device.stream);
    const std::string label = "ple block T=" + std::to_string(tokens) + (graph ? " graph" : "");
    int failures =
        verify_reduction(label + " residual", from_device_bf16(d_residual, wide), expected, kBlock);
    failures +=
        verify_reduction(label + " history", from_device_bf16(d_state, expected_state.size()),
                         expected_state, kBlock);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        for (const std::int32_t tokens : {1, 5, 256}) {
            failures += gate_case(tokens, 0x10U + tokens);
        }
        failures += conv_case(1, 1, 0x100U);
        failures += conv_case(1, 2, 0x200U);
        failures += conv_case(8, 2, 0x300U);
        failures += conv_case(13, 1, 0x400U);
        failures += conv_case(4, 2, 0x500U, 0); // a cancelled row (commit count 0)
        failures += in_place_advance_case(0x500U);
        for (const std::int32_t tokens : {1, 4, 8, 17})
            failures += block_case(tokens, false, 0x600U + tokens);
        failures += block_case(4, true, 0x700U);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " ple correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ple: " << error.what() << '\n';
        return 1;
    }
}
