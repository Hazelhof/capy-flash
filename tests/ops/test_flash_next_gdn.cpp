// Qwen3.8-Flash-Next Gated DeltaNet mixer composed from the tp2 Ops, after the projections:
//
//   gdn_projected_conv_{snapshot,record} -> gdn_gating -> gated_delta_net_{batch_update,
//   replay_record} -> gated_rmsnorm_sigmoid, then gdn_replay_fold over the 36-layer geometry.
//
// Ports the intent of ninfer6000 tests/ops/test_flash_next_gdn.cpp (FP64 oracle of the complete
// block with nonzero persistent state, selected slots and a CUDA Graph replay) to the tp2 ReplaySSM
// contract: MTP verify windows record their transitions, a random accepted prefix is folded into
// the destination slot, ordinary decode updates selected slots inside a captured graph. Every round
// is judged against an FP64 oracle evaluated from the represented device state the round started
// from, so each round is one bounded-length qualification. It runs at the full 16/48-head geometry
// and at one rank's 8/24-head half of a two-device split.
//
// Oracle boundaries: the BF16 q/k/v convolution outputs and the BF16 recurrent output are public
// tensors between Ops, so the oracle rounds them to BF16 (RNE) and is FP64 otherwise; g/beta are
// FP32 public tensors. Criteria: output = recurrent-output criterion + gated_rmsnorm criterion
// (their composition), state = the recurrent FP32 state criterion; conv history is exact.
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_replay.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ops/gdn_criteria.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim        = 128;
constexpr int kLayers     = 36;
constexpr int kLayer      = 7; // the layer the chain runs on; the fold covers all 36
constexpr int kRows       = 2;
constexpr int kSlots      = 6;
constexpr double kNormEps = 1.0e-6;

// The projected convolution keeps causal_conv1d_silu's single A16 criterion.
constexpr ReductionCriterion kConvCriterion{1.85e-3, 1.0e-3, 3.7e-3};
// Composition of the recurrent output criterion and gated_rmsnorm's criterion.
constexpr ReductionCriterion kOutputCriterion{6.0e-3, 5.0e-5, 8.5e-3};

struct Geometry {
    const char* name;
    int qk_heads;
    int value_heads;

    int query_rows() const { return qk_heads * kDim; }

    int value_rows() const { return value_heads * kDim; }

    int channels() const { return 2 * query_rows() + value_rows(); }
};

float hashed(std::uint64_t key, float lo, float hi) {
    key += 0x9e3779b97f4a7c15ULL;
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
    key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
    key ^= key >> 31;
    const double unit = static_cast<double>(key >> 11) * 0x1.0p-53;
    return static_cast<float>(lo + (hi - lo) * unit);
}

float bf16r(double x) { return bf16_to_f32(f32_to_bf16(static_cast<float>(x))); }

std::vector<float> random_bf16(std::size_t n, std::uint64_t seed, float lo, float hi) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) { v[i] = bf16r(hashed(seed * 1000003ULL + i, lo, hi)); }
    return v;
}

std::vector<std::uint16_t> bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = f32_to_bf16(v[i]); }
    return out;
}

double softplus(double x) { return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x))); }

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Params {
    std::vector<float> conv_weight; // [C,4], tap j of channel c at j*C+c
    std::vector<float> a_log, dt_bias;
    std::vector<float> norm;
};

// One row of one round: the represented inputs.
struct RowInputs {
    std::vector<float> projected; // [C,W]
    std::vector<float> z;         // [Hv*128,W]
    std::vector<float> a, b;      // [Hv,W]
};

struct RowOracle {
    std::vector<double> conv; // [C, valid] ideal SiLU(conv) before the BF16 boundary
    std::vector<double> y;    // [Hv*128, W]
    std::vector<std::vector<double>> state_after; // after t+1 tokens: [128,128,Hv]
};

// FP64 oracle of one row over `valid` tokens from the represented initial history/state.
RowOracle oracle_row(const Geometry& g, const Params& p, const std::vector<float>& history,
                     const std::vector<float>& state, const RowInputs& in, int width, int valid) {
    const int C = g.channels(), Q = g.query_rows(), Hv = g.value_heads, Hk = g.qk_heads;
    RowOracle out;
    out.y.assign(static_cast<std::size_t>(Hv) * kDim * width, 0.0);
    std::vector<double> s(state.begin(), state.end());
    std::vector<double> u(history.begin(), history.end()); // [C,3] oldest first, channel-major
    for (int t = 0; t < valid; ++t) {
        // Convolution + SiLU, rounded at the BF16 q/k/v boundary.
        std::vector<double> conv(C);
        for (int c = 0; c < C; ++c) {
            const double x = in.projected[static_cast<std::size_t>(t) * C + c];
            double acc     = 0.0;
            for (int j = 0; j < 3; ++j) { acc += p.conv_weight[j * C + c] * u[j * C + c]; }
            acc += p.conv_weight[3 * C + c] * x;
            const double activated = acc / (1.0 + std::exp(-acc));
            out.conv.push_back(activated);
            conv[c]      = bf16r(activated);
            u[0 * C + c] = u[1 * C + c];
            u[1 * C + c] = u[2 * C + c];
            u[2 * C + c] = x;
        }
        std::vector<double> q(conv.begin(), conv.begin() + Q),
            k(conv.begin() + Q, conv.begin() + 2 * Q), v(conv.begin() + 2 * Q, conv.end());
        for (int h = 0; h < Hk; ++h) {
            double qs = 0.0, ks = 0.0;
            for (int d = 0; d < kDim; ++d) {
                qs += q[h * kDim + d] * q[h * kDim + d];
                ks += k[h * kDim + d] * k[h * kDim + d];
            }
            for (int d = 0; d < kDim; ++d) {
                q[h * kDim + d] /= std::sqrt(qs + kNormEps);
                k[h * kDim + d] /= std::sqrt(ks + kNormEps);
            }
        }
        std::vector<double> o(static_cast<std::size_t>(Hv) * kDim);
        for (int h = 0; h < Hv; ++h) {
            const int qh      = h / (Hv / Hk);
            const double gate = static_cast<float>(
                -std::exp(static_cast<double>(p.a_log[h])) *
                softplus(static_cast<double>(in.a[static_cast<std::size_t>(t) * Hv + h]) +
                         p.dt_bias[h]));
            const double beta = static_cast<float>(
                sigmoid(static_cast<double>(in.b[static_cast<std::size_t>(t) * Hv + h])));
            const double alpha = std::exp(gate);
            double* S          = s.data() + static_cast<std::size_t>(h) * kDim * kDim;
            std::vector<double> delta(kDim);
            for (int r = 0; r < kDim; ++r) {
                double sk = 0.0;
                for (int c = 0; c < kDim; ++c) { sk += S[r * kDim + c] * k[qh * kDim + c]; }
                delta[r] = beta * (v[h * kDim + r] - alpha * sk);
            }
            for (int r = 0; r < kDim; ++r) {
                for (int c = 0; c < kDim; ++c) {
                    S[r * kDim + c] = alpha * S[r * kDim + c] + delta[r] * k[qh * kDim + c];
                }
            }
            for (int r = 0; r < kDim; ++r) {
                double sq = 0.0;
                for (int c = 0; c < kDim; ++c) { sq += S[r * kDim + c] * q[qh * kDim + c]; }
                o[static_cast<std::size_t>(h) * kDim + r] = bf16r(sq / std::sqrt(double(kDim)));
            }
        }
        out.state_after.push_back(s);
        // Sigmoid-gated RMSNorm per value head.
        for (int h = 0; h < Hv; ++h) {
            double ss = 0.0;
            for (int d = 0; d < kDim; ++d) { ss += o[h * kDim + d] * o[h * kDim + d]; }
            const double inv = 1.0 / std::sqrt(ss / kDim + kNormEps);
            for (int d = 0; d < kDim; ++d) {
                const std::size_t at = (static_cast<std::size_t>(t) * Hv + h) * kDim + d;
                out.y[at]            = o[h * kDim + d] * inv * p.norm[d] * sigmoid(in.z[at]);
            }
        }
    }
    return out;
}

struct Fixture {
    Geometry g;
    Params p;
    DeviceBuffer conv_weight, a_log, dt_bias, norm;
    DeviceBuffer state_storage;
    LinearAttentionStatePoolLayout state_layout;
    std::unique_ptr<LinearAttentionStatePool> pool;

    explicit Fixture(Geometry geometry) : g(geometry) {
        const int C = g.channels(), Hv = g.value_heads;
        p.conv_weight = random_bf16(static_cast<std::size_t>(C) * 4, 11, -0.6F, 0.6F);
        p.a_log.resize(Hv);
        p.dt_bias.resize(Hv);
        for (int h = 0; h < Hv; ++h) {
            p.a_log[h]   = hashed(1200 + h, -1.0F, 1.5F);
            p.dt_bias[h] = hashed(1300 + h, -1.0F, 1.0F);
        }
        p.norm      = random_bf16(kDim, 13, 0.5F, 1.5F);
        conv_weight = to_device(bits(p.conv_weight));
        a_log       = to_device(p.a_log);
        dt_bias     = to_device(p.dt_bias);
        norm        = to_device(bits(p.norm));

        LayoutBuilder state_builder;
        state_layout = plan_linear_attention_state_pool(
            state_builder, {.layers         = static_cast<std::uint32_t>(kLayers),
                            .conv_channels  = C,
                            .conv_width     = 3,
                            .value_heads    = Hv,
                            .value_head_dim = kDim,
                            .key_head_dim   = kDim,
                            .slot_count     = kSlots,
                            .conv_dtype     = DType::BF16});
        const std::size_t state_bytes = state_builder.finish(256);
        state_storage                 = DeviceBuffer(state_bytes);
        state_storage.fill(0);
        pool = std::make_unique<LinearAttentionStatePool>(DeviceSpan{state_storage.p, state_bytes},
                                                          state_layout);
        // Nonzero initial history and state in every slot of the chained layer.
        for (int slot = 0; slot < kSlots; ++slot) {
            const auto history =
                random_bf16(static_cast<std::size_t>(C) * 3, 100 + slot, -1.0F, 1.0F);
            const auto h_bits = bits(history);
            const Tensor conv = pool->conv_slot(kLayer, slot);
            cuda_check(
                cudaMemcpy(conv.data, h_bits.data(), h_bits.size() * 2, cudaMemcpyHostToDevice),
                "upload history");
            std::vector<float> state(static_cast<std::size_t>(kDim) * kDim * Hv);
            for (std::size_t i = 0; i < state.size(); ++i) {
                state[i] = hashed(200000ULL * (slot + 1) + i, -0.05F, 0.05F);
            }
            const Tensor rec = pool->recurrent_slot(kLayer, slot);
            cuda_check(cudaMemcpy(rec.data, state.data(), state.size() * 4, cudaMemcpyHostToDevice),
                       "upload state");
        }
    }

    std::vector<float> history(int slot) const {
        const Tensor conv = pool->conv_slot(kLayer, slot);
        const auto raw    = from_device<std::uint16_t>(conv.data, conv.numel());
        std::vector<float> out(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) { out[i] = bf16_to_f32(raw[i]); }
        return out;
    }

    std::vector<float> state(int slot) const {
        const Tensor rec = pool->recurrent_slot(kLayer, slot);
        return from_device<float>(rec.data, rec.numel());
    }
};

// ReplaySSM records of one verify width (the Program fixes K+1 per family), all 36 layers.
struct Records {
    DeviceBuffer storage;
    GdnReplayRecordLayout layout;
    GdnReplayRecords records;
    std::unique_ptr<ops::GdnReplayFoldPlan> fold;

    Records(const Geometry& g, int width, const LinearAttentionStatePool& pool) {
        LayoutBuilder builder;
        layout                  = plan_gdn_replay_records(builder, {.layers          = kLayers,
                                                                    .record_capacity = kRows,
                                                                    .width           = width,
                                                                    .conv_channels   = g.channels(),
                                                                    .qk_heads        = g.qk_heads,
                                                                    .value_heads     = g.value_heads,
                                                                    .key_dim         = kDim,
                                                                    .value_dim       = kDim});
        const std::size_t bytes = builder.finish(256);
        storage                 = DeviceBuffer(bytes);
        storage.fill(0); // other layers fold zero records: alpha=1, beta=0 keeps their state
        records = GdnReplayRecords({storage.p, bytes}, layout);
        fold    = std::make_unique<ops::GdnReplayFoldPlan>(records, pool.all_layers_view());
    }
};

RowInputs make_inputs(const Geometry& g, int width, std::uint64_t seed) {
    RowInputs in;
    in.projected = random_bf16(static_cast<std::size_t>(g.channels()) * width, seed, -1.5F, 1.5F);
    in.z = random_bf16(static_cast<std::size_t>(g.value_rows()) * width, seed + 1, -3.0F, 3.0F);
    in.a = random_bf16(static_cast<std::size_t>(g.value_heads) * width, seed + 2, -3.0F, 3.0F);
    in.b = random_bf16(static_cast<std::size_t>(g.value_heads) * width, seed + 3, -3.0F, 3.0F);
    return in;
}

// Device buffers of one round, laid out [.., W, B].
struct RoundBuffers {
    int width;
    DeviceBuffer projected, z, a, b, q, k, v, gg, beta, o, y, valid, src, dst;

    RoundBuffers(const Geometry& g, int w)
        : width(w), projected(static_cast<std::size_t>(g.channels()) * w * kRows * 2),
          z(static_cast<std::size_t>(g.value_rows()) * w * kRows * 2),
          a(static_cast<std::size_t>(g.value_heads) * w * kRows * 2),
          b(static_cast<std::size_t>(g.value_heads) * w * kRows * 2),
          q(static_cast<std::size_t>(g.query_rows()) * w * kRows * 2),
          k(static_cast<std::size_t>(g.query_rows()) * w * kRows * 2),
          v(static_cast<std::size_t>(g.value_rows()) * w * kRows * 2),
          gg(static_cast<std::size_t>(g.value_heads) * w * kRows * 4),
          beta(static_cast<std::size_t>(g.value_heads) * w * kRows * 4),
          o(static_cast<std::size_t>(g.value_rows()) * w * kRows * 2),
          y(static_cast<std::size_t>(g.value_rows()) * w * kRows * 2), valid(kRows * 4),
          src(kRows * 4), dst(kRows * 4) {}

    void upload(const std::vector<RowInputs>& rows) {
        const auto put = [&](DeviceBuffer& buffer, auto member) {
            std::vector<std::uint16_t> all;
            for (const RowInputs& r : rows) {
                const auto b16 = bits(r.*member);
                all.insert(all.end(), b16.begin(), b16.end());
            }
            buffer.copy_from_host(all.data(), all.size() * 2);
        };
        put(projected, &RowInputs::projected);
        put(z, &RowInputs::z);
        put(a, &RowInputs::a);
        put(b, &RowInputs::b);
    }
};

// Ordinary decode / gating / recurrence / norm half shared by both routes.
void gate_and_norm_launch(const Geometry& g, Fixture& f, RoundBuffers& r, cudaStream_t stream) {
    const int columns = r.width * kRows;
    Tensor gg(r.gg.p, DType::FP32, {g.value_heads, columns});
    Tensor beta(r.beta.p, DType::FP32, {g.value_heads, columns});
    ops::gdn_gating(Tensor(r.a.p, DType::BF16, {g.value_heads, columns}),
                    Tensor(r.b.p, DType::BF16, {g.value_heads, columns}),
                    Tensor(f.a_log.p, DType::FP32, {g.value_heads}),
                    Tensor(f.dt_bias.p, DType::FP32, {g.value_heads}), gg, beta, stream);
}

void norm_launch(const Geometry& g, Fixture& f, RoundBuffers& r, cudaStream_t stream) {
    const int columns = r.width * kRows;
    Tensor y(r.y.p, DType::BF16, {kDim, g.value_heads, columns});
    ops::gated_rmsnorm_sigmoid(Tensor(r.o.p, DType::BF16, {kDim, g.value_heads, columns}),
                               Tensor(f.norm.p, DType::BF16, {kDim}),
                               Tensor(r.z.p, DType::BF16, {kDim, g.value_heads, columns}),
                               static_cast<float>(kNormEps), y, stream);
}

constexpr float kScale = 0.08838834764831845F; // 1/sqrt(128)

void decode_round(const Geometry& g, Fixture& f, RoundBuffers& r, cudaStream_t stream) {
    const auto layer = f.pool->layer_view(kLayer);
    Tensor conv      = layer.conv;
    Tensor q(r.q.p, DType::BF16, {g.query_rows(), 1, kRows});
    Tensor k(r.k.p, DType::BF16, {g.query_rows(), 1, kRows});
    Tensor v(r.v.p, DType::BF16, {g.value_rows(), 1, kRows});
    const Tensor src(r.src.p, DType::I32, {kRows}), dst(r.dst.p, DType::I32, {kRows});
    ops::gdn_projected_conv_snapshot(Tensor(r.projected.p, DType::BF16, {g.channels(), 1, kRows}),
                                     Tensor(f.conv_weight.p, DType::BF16, {g.channels(), 4}), conv,
                                     Tensor{}, src, dst, q, k, v, stream);
    gate_and_norm_launch(g, f, r, stream);
    Tensor recurrent = layer.recurrent;
    Tensor o(r.o.p, DType::BF16, {kDim, g.value_heads, 1, kRows});
    ops::gated_delta_net_batch_update(Tensor(r.q.p, DType::BF16, {kDim, g.qk_heads, 1, kRows}),
                                      Tensor(r.k.p, DType::BF16, {kDim, g.qk_heads, 1, kRows}),
                                      Tensor(r.v.p, DType::BF16, {kDim, g.value_heads, 1, kRows}),
                                      Tensor(r.gg.p, DType::FP32, {g.value_heads, 1, kRows}),
                                      Tensor(r.beta.p, DType::FP32, {g.value_heads, 1, kRows}),
                                      kScale, true, recurrent, src, dst, o, stream);
    norm_launch(g, f, r, stream);
}

void verify_round(const Geometry& g, Fixture& f, Records& replay, RoundBuffers& r,
                  cudaStream_t stream) {
    const int W                  = r.width;
    const auto layer             = f.pool->layer_view(kLayer);
    GdnReplayRecordLayer records = replay.records.layer(kLayer, kRows);
    // The projection writes the conv record directly; here the represented projection is copied.
    Tensor record = records.conv;
    cuda_check(cudaMemcpyAsync(record.data, r.projected.p, record.bytes(), cudaMemcpyDeviceToDevice,
                               stream),
               "write conv record");
    Tensor q(r.q.p, DType::BF16, {g.query_rows(), W, kRows});
    Tensor k(r.k.p, DType::BF16, {g.query_rows(), W, kRows});
    Tensor v(r.v.p, DType::BF16, {g.value_rows(), W, kRows});
    const Tensor valid(r.valid.p, DType::I32, {kRows}), src(r.src.p, DType::I32, {kRows});
    ops::gdn_projected_conv_record(record, Tensor(f.conv_weight.p, DType::BF16, {g.channels(), 4}),
                                   layer.conv, valid, src, q, k, v, stream);
    gate_and_norm_launch(g, f, r, stream);
    Tensor o(r.o.p, DType::BF16, {kDim, g.value_heads, W, kRows});
    Tensor key_record   = records.key;
    Tensor value_record = records.value;
    Tensor gate_record  = records.gate;
    ops::gated_delta_net_replay_record(Tensor(r.q.p, DType::BF16, {kDim, g.qk_heads, W, kRows}),
                                       Tensor(r.k.p, DType::BF16, {kDim, g.qk_heads, W, kRows}),
                                       Tensor(r.v.p, DType::BF16, {kDim, g.value_heads, W, kRows}),
                                       Tensor(r.gg.p, DType::FP32, {g.value_heads, W, kRows}),
                                       Tensor(r.beta.p, DType::FP32, {g.value_heads, W, kRows}),
                                       kScale, layer.recurrent, valid, src, key_record,
                                       value_record, gate_record, o, stream);
    norm_launch(g, f, r, stream);
}

int check_round(const Geometry& g, Fixture& f, RoundBuffers& r, const std::vector<RowInputs>& in,
                const std::vector<RowOracle>& oracle, const std::vector<int>& valid,
                const std::string& label) {
    int failures = 0;
    const int W  = r.width;
    const auto y = from_device_bf16(r.y.p, static_cast<std::size_t>(g.value_rows()) * W * kRows);
    const int Q = g.query_rows(), V = g.value_rows();
    const auto q = from_device_bf16(r.q.p, static_cast<std::size_t>(Q) * W * kRows);
    const auto k = from_device_bf16(r.k.p, static_cast<std::size_t>(Q) * W * kRows);
    const auto v = from_device_bf16(r.v.p, static_cast<std::size_t>(V) * W * kRows);
    for (int row = 0; row < kRows; ++row) {
        std::vector<double> conv;
        for (int t = 0; t < valid[row]; ++t) {
            const std::size_t column = static_cast<std::size_t>(row) * W + t;
            conv.insert(conv.end(), q.begin() + column * Q, q.begin() + (column + 1) * Q);
            conv.insert(conv.end(), k.begin() + column * Q, k.begin() + (column + 1) * Q);
            conv.insert(conv.end(), v.begin() + column * V, v.begin() + (column + 1) * V);
        }
        failures += verify_reduction(label + " row " + std::to_string(row) + " conv", conv,
                                     oracle[row].conv, kConvCriterion);
    }
    for (int row = 0; row < kRows; ++row) {
        const std::size_t span = static_cast<std::size_t>(g.value_rows()) * W;
        std::vector<double> got(y.begin() + static_cast<std::ptrdiff_t>(row * span),
                                y.begin() + static_cast<std::ptrdiff_t>(row * span) +
                                    static_cast<std::ptrdiff_t>(g.value_rows()) * valid[row]);
        std::vector<double> ref(oracle[row].y.begin(),
                                oracle[row].y.begin() +
                                    static_cast<std::ptrdiff_t>(g.value_rows()) * valid[row]);
        failures += verify_reduction(label + " row " + std::to_string(row) + " output", got, ref,
                                     kOutputCriterion);
        for (std::size_t i = static_cast<std::size_t>(g.value_rows()) * valid[row]; i < span; ++i) {
            if (y[row * span + i] != 0.0) {
                std::cerr << label << " row " << row << ": invalid tail output is not zero\n";
                ++failures;
                break;
            }
        }
    }
    (void)in;
    return failures;
}

int check_state(const Geometry& g, Fixture& f, int slot, const std::vector<double>& expected,
                const std::vector<float>& history_before, const std::vector<float>& projected,
                int committed, const std::string& label) {
    const int C      = g.channels();
    const auto state = f.state(slot);
    int failures =
        verify_reduction(label + " state", std::vector<double>(state.begin(), state.end()),
                         expected, gated_delta_net_state_fp32_criterion());
    // History = tail_3(old || committed projected columns), exactly.
    std::vector<float> want(static_cast<std::size_t>(C) * 3);
    for (int c = 0; c < C; ++c) {
        std::vector<float> seq = {history_before[c], history_before[C + c],
                                  history_before[2 * C + c]};
        for (int t = 0; t < committed; ++t) {
            seq.push_back(projected[static_cast<std::size_t>(t) * C + c]);
        }
        for (int j = 0; j < 3; ++j) {
            want[static_cast<std::size_t>(j) * C + c] = seq[seq.size() - 3 + j];
        }
    }
    failures += verify_exact((label + " conv history").c_str(), f.history(slot), want);
    return failures;
}

int run_geometry(const Geometry& g) {
    int failures = 0;
    Fixture f(g);
    std::map<int, std::unique_ptr<Records>> replay; // per verify width
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");

    // Each row alternates between two private slots.
    int slot_a[kRows] = {0, 3}, slot_b[kRows] = {1, 4};
    RoundBuffers decode(g, 1);
    // Ordinary decode captured once; replays read slot selectors and inputs from device memory.
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;

    struct Plan {
        int width;               // 1 = ordinary decode, otherwise MTP verify
        std::vector<int> valid;  // per row
        std::vector<int> commit; // per row (verify)
    };

    const std::vector<Plan> plans = {
        {4, {4, 4}, {3, 1}}, {1, {1, 1}, {}}, {8, {8, 5}, {8, 0}}, {1, {1, 1}, {}},
        {2, {2, 1}, {2, 1}}, {1, {1, 1}, {}}, {6, {6, 6}, {0, 6}},
    };
    std::uint64_t seed = 5000;
    for (std::size_t round = 0; round < plans.size(); ++round) {
        const Plan& plan        = plans[round];
        const int W             = plan.width;
        const std::string label = std::string("flash-next GDN ") + g.name + " round " +
                                  std::to_string(round) +
                                  (W == 1 ? " decode" : " verify W=" + std::to_string(W));
        std::vector<RowInputs> inputs;
        std::vector<RowOracle> oracles;
        std::vector<std::vector<float>> hist_before, state_before;
        for (int row = 0; row < kRows; ++row) {
            inputs.push_back(make_inputs(g, W, seed += 10));
            hist_before.push_back(f.history(slot_a[row]));
            state_before.push_back(f.state(slot_a[row]));
            oracles.push_back(oracle_row(g, f.p, hist_before[row], state_before[row], inputs[row],
                                         W, plan.valid[row]));
        }
        const std::vector<int> src = {slot_a[0], slot_a[1]}, dst = {slot_b[0], slot_b[1]};
        if (W == 1) {
            decode.upload(inputs);
            decode.src.copy_from_host(src.data(), src.size() * 4);
            decode.dst.copy_from_host(dst.data(), dst.size() * 4);
            if (executable == nullptr) {
                decode_round(g, f, decode, stream); // eager warm-up of the same round
                cuda_synchronize(stream);
                // The round reads its source slots and writes distinct destinations, so the
                // captured replay below recomputes the same transition.
                failures +=
                    check_round(g, f, decode, inputs, oracles, plan.valid, label + " eager");
                cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                           "begin capture");
                decode_round(g, f, decode, stream);
                cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
                cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                           "instantiate");
            }
            cuda_check(cudaGraphLaunch(executable, stream), "replay decode");
            cuda_synchronize(stream);
            failures += check_round(g, f, decode, inputs, oracles, plan.valid, label + " graph");
            for (int row = 0; row < kRows; ++row) {
                failures +=
                    check_state(g, f, slot_b[row], oracles[row].state_after[0], hist_before[row],
                                inputs[row].projected, 1, label + " row " + std::to_string(row));
                std::swap(slot_a[row], slot_b[row]);
            }
            continue;
        }
        RoundBuffers buffers(g, W);
        buffers.upload(inputs);
        buffers.valid.copy_from_host(plan.valid.data(), plan.valid.size() * 4);
        buffers.src.copy_from_host(src.data(), src.size() * 4);
        auto& records = replay[W];
        if (!records) { records = std::make_unique<Records>(g, W, *f.pool); }
        verify_round(g, f, *records, buffers, stream);
        cuda_synchronize(stream);
        failures += check_round(g, f, buffers, inputs, oracles, plan.valid, label);
        // The source slots are unchanged by the verify; commit the accepted prefixes.
        std::vector<ops::GdnReplayFoldRow> rows;
        for (int row = 0; row < kRows; ++row) {
            rows.push_back({slot_a[row], slot_b[row], plan.commit[row]});
        }
        records->fold->execute(rows, stream);
        cuda_synchronize(stream);
        for (int row = 0; row < kRows; ++row) {
            const int committed = plan.commit[row];
            const std::string row_label =
                label + " row " + std::to_string(row) + " commit " + std::to_string(committed);
            failures += verify_exact((row_label + " source state untouched").c_str(),
                                     f.state(slot_a[row]), state_before[row]);
            if (committed == 0) { continue; } // destination untouched, the source stays current
            failures += check_state(g, f, slot_b[row], oracles[row].state_after[committed - 1],
                                    hist_before[row], inputs[row].projected, committed, row_label);
            std::swap(slot_a[row], slot_b[row]);
        }
    }
    cuda_check(cudaGraphExecDestroy(executable), "destroy executable");
    cuda_check(cudaGraphDestroy(graph), "destroy graph");
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
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
        failures += run_geometry({"16/48", 16, 48});
        failures += run_geometry({"shard 8/24", 8, 24});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " flash_next_gdn\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "flash_next_gdn: " << error.what() << '\n';
        return 1;
    }
}
