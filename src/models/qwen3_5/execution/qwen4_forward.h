#pragma once

// Qwen4Exp (Qwen3.8-Flash-Next) Text forward at one device or at tensor-parallel width 2:
// hyper-connection residual (T10 Ops), QSA attention with its index planes and GDN mixers (T11
// Ops), PLE before block 1 (T10), the router + shared expert + routed experts (moe_route.h,
// qwen4_experts.h) and the output mixer. Mathematics: docs/maintainer/qwen4-exp-model.md.
// Execution order follows ninfer-ext's qwen4_text.cpp (giveen/ninfer-ext @9c933f01, Apache-2.0)
// with this tree's Ops.
//
// One template serves execution (Arena = WorkspaceArena: allocates and launches) and startup
// sizing (Arena = WorkspaceLayoutBuilder: the same allocations in the same scopes, each Op
// replaced by its declared scratch), so the planned workspace is the executed one by
// construction. In planning mode T is the interval's largest column count and every Linear
// reserves its capacity over [first, last].
//
// Tensor parallelism (width 2, load/sharding.h placements): every rank holds the whole wide
// residual (the hyper-connection mixers, the router, the QSA indexer and the PLE are replicated)
// and issues the same replicated work on its own stream; the per-rank shards are the QSA heads
// (12 query + 1 KV), the GDN heads (8 key + 24 value), the vocabulary half of the output head,
// the routed experts of its ID parity and the shared expert of its blocks (block % 2). Each block
// has exactly two exchanges:
//   * the mixer output: each rank's row-parallel output projection, one BF16 all-reduce
//     (ops::linear_row_parallel), so both ranks hold the identical BF16 y;
//   * the MoE: each rank's FP32 partial (its routed experts from its Qwen4RoutedExperts backend,
//     plus on the owner rank the gated shared expert, ops::moe_accumulate_shared), one FP32
//     all-reduce rounded to BF16 once (ops::allreduce_sum_f32).
// The router reads identical bits on both ranks (deterministic replicated kernels), so both ranks
// route every token to the same experts and the partition by ID parity is exact. At width 1 the
// schedule is the single-device one, unchanged.

#include "core/arena.h"
#include "core/device.h"
#include "core/device_scope.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/nvtx.h"
#include "core/tensor.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/qwen4_experts.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/copy_rows.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/moe_grouped.h"
#include "ninfer/ops/moe_route.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/sparse_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::models::qwen3_5::execution {

// Host taps of the wide residual after selected layers (teacher-forcing dumps, eager only).
class Qwen4Taps {
public:
    virtual ~Qwen4Taps() = default;
    [[nodiscard]] virtual bool wants(std::int32_t layer) const = 0;
    // `wide` BF16 [S*H, T] committed residual after `layer` (or the head mixer output when
    // layer == -1, BF16 [H,T]); column t is sequence position first_position + t.
    virtual void capture(std::int32_t layer, const Tensor& wide, std::int32_t first_position,
                         cudaStream_t stream) = 0;
    // tp 2: the same tap's two rank copies (each on its rank and stream), which the replicated
    // schedule keeps bit-identical (identical routing on both ranks depends on it).
    virtual void compare_ranks(std::int32_t layer, const Tensor& rank0, const Tensor& rank1,
                               cudaStream_t stream0, cudaStream_t stream1) {
        (void)layer, (void)rank0, (void)rank1, (void)stream0, (void)stream1;
    }
    // T128: the two rank partials of a TP2 prefill exchange, before the sum (eager only).
    // site 0 = mixer output projection (BF16 [H,T]), 1 = MoE (FP32 [H,T], NINFER_TP_AR=0 path).
    [[nodiscard]] virtual bool wants_exchange() const { return false; }
    virtual void capture_exchange(std::int32_t layer, int site, const Tensor& rank0,
                                  const Tensor& rank1, std::int32_t first_position,
                                  cudaStream_t stream0, cudaStream_t stream1) {
        (void)layer, (void)site, (void)rank0, (void)rank1, (void)first_position, (void)stream0,
            (void)stream1;
    }
    // MTP teacher forcing (T24): every prompt column runs the whole MTP layer and its draft
    // logits (BF16 [V,1] of the column at `position`, predicting the token at position + 2) are
    // captured, with the prompt tokens.
    [[nodiscard]] virtual bool wants_mtp_logits() const { return false; }
    virtual void capture_mtp_logits(const Tensor&, std::int32_t, cudaStream_t) {}
    virtual void capture_tokens(std::span<const std::int32_t>) {}
};

// Host probe of a routed-expert call's inputs (eager validation only; T24 calibrates the MTP
// layer's activation ranges with it): x BF16 [H,T], ids I32 [10,T], weights FP32 [10,T].
class Qwen4MoeProbe {
public:
    virtual ~Qwen4MoeProbe() = default;
    [[nodiscard]] virtual bool wants(std::int32_t layer) const = 0;
    virtual void capture(std::int32_t layer, const Tensor& x, const Tensor& ids,
                         const Tensor& weights, cudaStream_t stream) = 0;
};

struct Qwen4DraftHead; // qwen4_draft_head.h

// Program-owned inputs a Qwen4Exp TextContext reads (bound per execution core). At tp 2 each rank
// has its own: rank 0's is the ExecutionCore's, rank 1's TpExecution::qwen4. The PLE gather, its
// staging and the decode host rows are rank 0's (one host gather, uploaded to both ranks); the
// taps are rank 0's alone (the replicated residual).
struct Qwen4Runtime {
    // This rank's routed-expert backend: at tp 2 it computes the experts of the rank's ID parity.
    Qwen4RoutedExperts* experts = nullptr;
    // MTP layer (T24): its routed-expert backend and its bank with per-expert divisor planes (the
    // Program expands a bank-wide divisor; Qwen4MtpParameters). The backend may be `experts`
    // itself (the offload backend serves the MTP layer as layer num_hidden_layers). Null without
    // MTP.
    Qwen4RoutedExperts* mtp_experts        = nullptr;
    const Qwen4ExpertBank* mtp_expert_bank = nullptr;
    // PLE convolution states of the PLE layer, BF16 [rows, slots] (StateImage).
    Tensor ple_states;
    // Speculative verification: the PLE convolution inputs of the round, BF16 [rows, K+1,
    // max_concurrency]; the commit advances the history by each row's accepted prefix (T10's
    // record/fold contract). Empty without a speculative backend.
    Tensor ple_record;
    // Prefill: host n-gram gather and pinned BF16 staging [embed_dim, prefill_chunk].
    PleGather* ple_gather   = nullptr;
    std::uint16_t* ple_host = nullptr;
    // Decode rounds: BF16 [embed_dim, columns] rows of the round's tokens (ordinary: one column per
    // lane; MTP: K+1 columns per lane, lane-major), uploaded by the round from `decode_ple_host`
    // (pinned, written by the Program before the launch).
    Tensor decode_ple_rows;
    const std::uint16_t* decode_ple_host = nullptr;
    Qwen4Taps* taps = nullptr;
    Qwen4MoeProbe* moe_probe = nullptr;
    // T43: the MTP proposal head over a draft token shortlist (rank 0's at tp 2), null = the full
    // Text output head. Program-owned (qwen4_draft_head.h).
    const Qwen4DraftHead* draft_head = nullptr;
};

// The columns one forward processes: W columns of B sequences (prefill: W = T, B = 1).
struct Qwen4Columns {
    std::int32_t width = 0;
    std::int32_t batch = 0;
    const Tensor* cache_positions = nullptr; // I32 [T]
    const Tensor* rope_positions  = nullptr; // I32 [T] or [T,3] (axis-major)
    const Tensor* valid_columns   = nullptr; // null or I32 [B]
    // MTP decode (T52): the MoE routes only each lane's last valid column (the alignment of a
    // verified window is read only at the accepted column, ops::MoeRouteColumns::LastValid).
    bool moe_last_valid_only      = false;
    const Tensor* table_rows      = nullptr; // I32 [B]
    std::uint32_t max_visible     = 0;       // envelope: largest visible extent of any column
    bool prefill                  = true;    // prefill: scalar state slots; else [B] selectors
    std::int32_t source_slot      = 0;
    std::int32_t destination_slot = 0;
    const Tensor* source_slots      = nullptr; // I32 [B]
    const Tensor* destination_slots = nullptr; // I32 [B]
    std::int32_t first_position     = 0;       // absolute position of column 0 (taps)
    // Speculative verification (W = K+1 columns per lane): GDN writes ReplaySSM records instead
    // of advancing its state and PLE copies its convolution inputs to Qwen4Runtime::ple_record;
    // the commit folds the accepted prefix. QSA K/V and index planes are appended in place and
    // rejected columns are overwritten by the next round (position-ordered re-append).
    const GdnReplayRecords* replay_records = nullptr;
};

struct Qwen4State {
    const PagedKVCache* kv              = nullptr;
    LinearAttentionStatePool* linear    = nullptr;
    const Qwen4Runtime* runtime         = nullptr;
};

[[nodiscard]] inline ops::QsaGeometry qwen4_qsa_geometry(const TextConfig& config) {
    const auto& attention = *config.attention;
    const auto& qsa       = *config.sparse_attention;
    const auto& rope      = *config.rope_parameters;
    ops::QsaGeometry g;
    g.query_heads  = dimension(attention.num_attention_heads);
    g.kv_heads     = dimension(attention.num_key_value_heads);
    g.head_dim     = dimension(attention.head_dim);
    g.index_heads  = dimension(qsa.indexer_heads);
    g.index_dim    = dimension(qsa.indexer_head_dim);
    g.budget       = dimension(qsa.budget);
    g.ratio        = dimension(qsa.compress_ratio);
    g.rotary_dim   = dimension(rope.rotary_dim);
    g.rope_theta   = rope.rope_theta;
    g.rms_norm_eps = config.rms_norm_eps;
    if (rope.pair_axes.size() > g.pair_axes.size()) {
        throw std::invalid_argument("QSA rotary pairs exceed the indexer geometry");
    }
    std::copy(rope.pair_axes.begin(), rope.pair_axes.end(), g.pair_axes.begin());
    return g;
}

// One rank's operands of a Qwen4Forward. `config` is the rank's geometry (shard_text_config at
// tp 2, the logical config at one device); `device` is only read at tp 2.
template <class Arena>
struct Qwen4Rank {
    const TextConfig* config     = nullptr;
    const Parameters* parameters = nullptr;
    Arena* work                  = nullptr;
    cudaStream_t stream          = nullptr;
    DeviceExecutionView execution{};
    int device = 0;
    Qwen4Columns columns;
    Qwen4State state;
};

// The two-device transport of an executing tp 2 forward.
struct Qwen4Collectives {
    const ExecutionContext* execution = nullptr;
    const ops::PeerEvents* events     = nullptr;
};

template <class Arena>
class Qwen4Forward {
public:
    static constexpr bool kExecute = std::is_same_v<Arena, WorkspaceArena>;
    using Rank        = Qwen4Rank<Arena>;
    using RankTensors = std::array<Tensor, 2>;

    // One device.
    Qwen4Forward(const TextConfig& config, const Parameters& parameters, Arena& work,
                 cudaStream_t stream, DeviceExecutionView execution, Qwen4Columns columns,
                 Qwen4State state, std::int32_t first, std::int32_t last)
        : Qwen4Forward(config,
                       std::array<Rank, 2>{Rank{.config     = &config,
                                                .parameters = &parameters,
                                                .work       = &work,
                                                .stream     = stream,
                                                .execution  = execution,
                                                .device     = 0,
                                                .columns    = columns,
                                                .state      = state},
                                           Rank{}},
                       1, {}, first, last) {}

    // `width` ranks of `ranks` (rank 0 first). `config` is the logical Text config; at width 2
    // `collectives` must be complete when executing (planning ignores it).
    Qwen4Forward(const TextConfig& config, std::array<Rank, 2> ranks, int width,
                 Qwen4Collectives collectives, std::int32_t first, std::int32_t last)
        : config_(config), r_(ranks), width_(width), tp_(collectives), first_(first),
          last_(last) {
        if (first_ <= 0 || last_ < first_) {
            throw std::invalid_argument("Qwen4Exp forward column interval is invalid");
        }
        if (width_ != 1 && width_ != 2) {
            throw std::invalid_argument("Qwen4Exp forward width must be 1 or 2");
        }
        if (r_[0].columns.replay_records != nullptr && r_[0].columns.prefill) {
            throw std::invalid_argument("Qwen4Exp speculative records require a decode window");
        }
        for (int r = 0; r < width_; ++r) {
            const Rank& rank = r_[static_cast<std::size_t>(r)];
            if (rank.config == nullptr || rank.parameters == nullptr || rank.work == nullptr ||
                !rank.parameters->qwen4 ||
                rank.parameters->qwen4->layers.size() != config_.layer_types.size()) {
                throw std::invalid_argument("Qwen4Exp forward rank " + std::to_string(r) +
                                            " has no Qwen4Exp Parameters");
            }
        }
        if constexpr (kExecute) {
            if (width_ == 2 &&
                (tp_.execution == nullptr || tp_.events == nullptr || tp_.execution->tp != 2)) {
                throw std::invalid_argument("Qwen4Exp tp 2 forward has no collectives");
            }
        }
        // The shared expert of each block: whole at one device, on exactly one rank at tp 2.
        for (std::size_t layer = 0; layer < config_.layer_types.size(); ++layer) {
            int holders = 0;
            for (int r = 0; r < width_; ++r) { holders += block(r, layer).moe.shared ? 1 : 0; }
            if (holders != 1) {
                throw std::invalid_argument("text/layers/" + std::to_string(layer) +
                                            ": the shared expert must be held by exactly one rank");
            }
        }
    }

    [[nodiscard]] std::int32_t wide_rows() const { return dimension(config_.residual_width()); }
    [[nodiscard]] std::int32_t hidden() const { return dimension(config_.hidden_size); }
    [[nodiscard]] std::int32_t streams() const {
        return dimension(config_.hyper_connection->hc_count);
    }

    // One device: ids I32 [T]; ple_embedding BF16 [embed_dim, T] (when the model has PLE);
    // wide_out BF16 [S*H, T] receives the committed residual after the last block: the carried
    // hidden of the round (the continuation hidden MTP consumes). head() turns it into the output
    // head's input.
    void run(const Tensor& ids, const Tensor* ple_embedding, Tensor& wide_out) {
        const RankTensors rank_ids{ids, Tensor{}};
        RankTensors rank_ple;
        if (ple_embedding != nullptr) { rank_ple = {*ple_embedding, Tensor{}}; }
        RankTensors rank_out{wide_out, Tensor{}};
        run(rank_ids, ple_embedding != nullptr ? &rank_ple : nullptr, rank_out);
    }

    // Per rank: ids, the PLE n-gram embedding and the wide output, each resident on its rank.
    // Both ranks' outputs hold the same (replicated) committed residual.
    void run(const RankTensors& ids, const RankTensors* ple_embedding, RankTensors& wide_out) {
        const std::int32_t T = columns();
        auto scope           = scopes();
        RankTensors wide     = wide_buffers(wide_out, T);
        {
            auto embed_scope = scopes();
            RankTensors e    = alloc(DType::BF16, {hidden(), T});
            each([&](int r) {
                if constexpr (kExecute) {
                    ops::embedding(ids[at(r)], parameters(r).text.token_embedding, e[at(r)], s(r));
                    ops::hyper_connection_expand(e[at(r)], streams(), wide[at(r)], s(r));
                }
            });
        }
        // The all-reduce staging of the two exchanges per block (tp 2 only).
        RankTensors staging_bf16;
        RankTensors staging_fp32;
        if (width_ == 2) {
            staging_bf16 = alloc(DType::BF16, {hidden(), T});
            staging_fp32 = alloc(DType::FP32, {hidden(), T});
        }
        RankTensors mixed = alloc(DType::BF16, {hidden(), T});
        std::array<RankTensors, 2> y{alloc(DType::BF16, {hidden(), T}),
                                     alloc(DType::BF16, {hidden(), T})};
        std::array<RankTensors, 2> inject{alloc(DType::FP32, {streams(), T}),
                                          alloc(DType::FP32, {streams(), T})};
        int pending       = -1;
        int next          = 0;
        const auto commit = [&] {
            if (pending < 0) { return; }
            each([&](int r) {
                if constexpr (kExecute) {
                    ops::hyper_connection_combine(y[at(pending)][at(r)], inject[at(pending)][at(r)],
                                                  wide[at(r)], s(r));
                }
            });
            pending = -1;
        };
        // Mix the residual into the next block input, committing the pending block output in
        // the same pass when there is one (T10's combine_mix).
        const auto mix = [&](HyperConnectionParameters Qwen4BlockParameters::* member,
                             std::size_t layer) {
            reserve(ops::hyper_connection_mix_workspace_bytes(first_, last_, true));
            each([&](int r) {
                if constexpr (kExecute) {
                    const HyperConnectionParameters& hc = block(r, layer).*member;
                    if (pending >= 0) {
                        ops::hyper_connection_combine_mix(
                            wide[at(r)], y[at(pending)][at(r)], inject[at(pending)][at(r)],
                            hc.weights, config_.rms_norm_eps, mixed[at(r)],
                            &inject[at(next)][at(r)], work(r), s(r));
                    } else {
                        ops::hyper_connection_mix(wide[at(r)], hc.weights, config_.rms_norm_eps,
                                                  mixed[at(r)], &inject[at(next)][at(r)], work(r),
                                                  s(r));
                    }
                }
            });
            pending = -1;
        };
        for (std::size_t layer = 0; layer < config_.layer_types.size(); ++layer) {
            const auto compact   = dimension(config_.compact_layer_indices[layer]);
            const bool attention = config_.layer_types[layer] == MixerKind::FullAttention;
            try {
                if (block(0, layer).ple) {
                    if (ple_embedding == nullptr) {
                        throw std::logic_error("Qwen4Exp PLE block has no n-gram embedding");
                    }
                    commit();
                    ple(layer, *ple_embedding, wide);
                }
                mix(&Qwen4BlockParameters::attention_hc, layer);
                if (attention) {
                    qsa(layer, mixed, &y[at(next)], static_cast<std::uint32_t>(compact),
                        staging_bf16);
                } else {
                    gdn(layer, mixed, y[at(next)], compact, staging_bf16);
                }
                pending = next;
                next ^= 1;
                mix(&Qwen4BlockParameters::ffn_hc, layer); // commits the mixer output in the same pass
                ahead(layer, wide);
                moe(layer, mixed, y[at(next)], staging_fp32);
                pending = next;
                next ^= 1;
                if constexpr (kExecute) {
                    const Qwen4Columns& c = r_[0].columns;
                    const Qwen4Runtime* runtime = r_[0].state.runtime;
                    if (c.prefill && runtime != nullptr && runtime->taps != nullptr &&
                        runtime->taps->wants(static_cast<std::int32_t>(layer))) {
                        commit();
                        std::optional<ScopedCurrentDevice> rank0;
                        if (width_ == 2) { rank0.emplace(r_[0].device); }
                        runtime->taps->capture(static_cast<std::int32_t>(layer), wide[0],
                                               c.first_position, s(0));
                        if (width_ == 2) {
                            runtime->taps->compare_ranks(static_cast<std::int32_t>(layer), wide[0],
                                                         wide[1], s(0), s(1));
                        }
                    }
                }
            } catch (const std::exception& error) {
                throw std::runtime_error("text/layers/" + std::to_string(layer) + " " + phase() +
                                         (width_ == 2 ? " tp2" : "") +
                                         " columns=" + std::to_string(T) + ": " + error.what());
            }
        }
        commit();
        if constexpr (kExecute) {
            const Qwen4Columns& c       = r_[0].columns;
            const Qwen4Runtime* runtime = r_[0].state.runtime;
            if (c.prefill && runtime != nullptr && runtime->taps != nullptr &&
                runtime->taps->wants(-1)) {
                auto tap_scope = scopes();
                RankTensors x  = alloc(DType::BF16, {hidden(), T});
                head(wide, x);
                std::optional<ScopedCurrentDevice> rank0;
                if (width_ == 2) { rank0.emplace(r_[0].device); }
                runtime->taps->capture(-1, x[0], c.first_position, s(0));
                if (width_ == 2) { runtime->taps->compare_ranks(-1, x[0], x[1], s(0), s(1)); }
            }
        }
    }

    // The Text output mixer over carried hiddens on every rank: wide BF16 [S*H, N] -> x BF16
    // [H, N], the output head's input (replicated at tp 2). N is the caller's column count.
    void head(const RankTensors& wide, RankTensors& x) {
        head({&qwen4(0).head, width_ == 2 ? &qwen4(1).head : nullptr}, wide, x);
    }

    // The same with the given output mixer per rank (the Text or the MTP layer's).
    void head(std::array<const HyperConnectionParameters*, 2> hc, const RankTensors& wide,
              RankTensors& x) {
        reserve(ops::hyper_connection_mix_workspace_bytes(first_, last_, false));
        each([&](int r) {
            if constexpr (kExecute) {
                ops::hyper_connection_mix(wide[at(r)], hc[at(r)]->weights, config_.rms_norm_eps,
                                          x[at(r)], nullptr, work(r), s(r));
            }
        });
    }

    // One device.
    void head(const HyperConnectionParameters& hc, const Tensor& wide, Tensor& x) {
        const RankTensors rank_wide{wide, Tensor{}};
        RankTensors rank_x{x, Tensor{}};
        head({&hc, nullptr}, rank_wide, rank_x);
    }

    // MTP layer (docs/maintainer/qwen4-exp-model.md#mtp) over T columns, with this forward's
    // state bound to the MTP KV cache (layer 0, QSA index planes) and the MTP expert backend:
    //   e   = embedding_projection . offset_rmsnorm(embed(ids), embedding_norm)
    //   h   = offset_rmsnorm(hidden, hidden_norm)                 (one RMS over S*H)
    //   R'  = [hidden_projection . h_s + e for each stream s]
    //   R'  = QSA_block(R')
    // ids I32 [T] (the shifted tokens), hidden BF16 [S*H, T] the target's carried hiddens,
    // embeddings optional BF16 [H, T] replacing embed(ids). `wide_out` BF16 [S*H, T] receives R'.
    // `append_only` stops after the block's QSA append (prefill KV of the prompt columns whose
    // proposals are not needed); `wide_out` is then scratch. Per rank at tp 2 (each rank's MTP
    // Parameters: its QSA heads, its expert partition, the shared expert on one rank).
    void run_mtp(std::array<const Qwen4MtpParameters*, 2> m, const RankTensors& ids,
                 const RankTensors* embeddings, const RankTensors& hidden_wide,
                 RankTensors& wide_out, bool append_only) {
        const std::int32_t T = columns();
        for (int r = 0; r < width_; ++r) {
            if (m[at(r)] == nullptr) {
                throw std::invalid_argument("Qwen4Exp MTP forward rank " + std::to_string(r) +
                                            " has no MTP Parameters");
            }
        }
        mtp_ = m;
        {
            int holders = 0;
            for (int r = 0; r < width_; ++r) { holders += m[at(r)]->layer.moe.shared ? 1 : 0; }
            if (holders != 1) {
                throw std::invalid_argument(
                    "mtp/layers/0: the shared expert must be held by exactly one rank");
            }
        }
        auto scope       = scopes();
        RankTensors wide = wide_buffers(wide_out, T);
        {
            auto stem_scope        = scopes();
            RankTensors e          = alloc(DType::BF16, {hidden(), T});
            RankTensors en         = alloc(DType::BF16, {hidden(), T});
            RankTensors projected  = alloc(DType::BF16, {hidden(), T});
            RankTensors hn         = alloc(DType::BF16, {wide_rows(), T});
            RankTensors streams_hp = alloc(DType::BF16, {wide_rows(), T});
            each([&](int r) {
                const std::size_t i         = at(r);
                const Qwen4MtpParameters& p = *m[i];
                if constexpr (kExecute) {
                    if (embeddings != nullptr) {
                        CUDA_CHECK(::ninfer::memcpy_device_async(e[i].data, (*embeddings)[i].data, e[i].bytes(), s(r)));
                    } else {
                        ops::embedding(ids[i], parameters(r).text.token_embedding, e[i], s(r));
                    }
                    ops::rmsnorm(e[i], p.embedding_norm, config_.rms_norm_eps, true, en[i], s(r));
                    ops::rmsnorm(hidden_wide[i], p.hidden_norm, config_.rms_norm_eps, true, hn[i],
                                 s(r));
                }
                linear(r, en[i], p.embedding_projection, projected[i]);
                // The [H,H] projection applies to every stream: [S*H, T] is [H, S*T] column-wise.
                if constexpr (kExecute) {
                    const Tensor hn_streams = hn[i].view({hidden(), streams() * T});
                    Tensor out_streams      = streams_hp[i].view({hidden(), streams() * T});
                    ops::linear(hn_streams, p.hidden_projection.weight, out_streams,
                                p.hidden_projection.policy, work(r), s(r));
                    ops::hyper_connection_expand(projected[i], streams(), wide[i], s(r));
                    ops::residual_add(streams_hp[i], wide[i], s(r));
                } else {
                    const LinearParameters& hp = p.hidden_projection;
                    reserve(ops::linear_workspace_capacity_bytes(hp.weight.qtype, hp.weight.n,
                                                                 hp.weight.k, hp.policy,
                                                                 first_ * streams(),
                                                                 last_ * streams()),
                            r);
                }
            });
        }
        const std::size_t layer = mtp_layer();
        RankTensors staging_bf16;
        RankTensors staging_fp32;
        if (width_ == 2) {
            staging_bf16 = alloc(DType::BF16, {hidden(), T});
            staging_fp32 = alloc(DType::FP32, {hidden(), T});
        }
        RankTensors mixed  = alloc(DType::BF16, {hidden(), T});
        RankTensors y      = alloc(DType::BF16, {hidden(), T});
        RankTensors inject = alloc(DType::FP32, {streams(), T});
        try {
            reserve(ops::hyper_connection_mix_workspace_bytes(first_, last_, true));
            each([&](int r) {
                if constexpr (kExecute) {
                    ops::hyper_connection_mix(wide[at(r)], block(r, layer).attention_hc.weights,
                                              config_.rms_norm_eps, mixed[at(r)], &inject[at(r)],
                                              work(r), s(r));
                }
            });
            qsa(layer, mixed, append_only ? nullptr : &y, 0, staging_bf16);
            if (append_only) { return; }
            each([&](int r) {
                if constexpr (kExecute) {
                    ops::hyper_connection_combine_mix(wide[at(r)], y[at(r)], inject[at(r)],
                                                      block(r, layer).ffn_hc.weights,
                                                      config_.rms_norm_eps, mixed[at(r)],
                                                      &inject[at(r)], work(r), s(r));
                }
            });
            moe(layer, mixed, y, staging_fp32);
            each([&](int r) {
                if constexpr (kExecute) {
                    ops::hyper_connection_combine(y[at(r)], inject[at(r)], wide[at(r)], s(r));
                }
            });
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("mtp/layers/0 ") + phase() +
                                     (width_ == 2 ? " tp2" : "") +
                                     " columns=" + std::to_string(T) + ": " + error.what());
        }
    }

    // One device.
    void run_mtp(const Qwen4MtpParameters& m, const Tensor& ids, const Tensor* embeddings,
                 const Tensor& hidden_wide, Tensor& wide_out, bool append_only) {
        const RankTensors rank_ids{ids, Tensor{}};
        RankTensors rank_embeddings;
        if (embeddings != nullptr) { rank_embeddings = {*embeddings, Tensor{}}; }
        const RankTensors rank_hidden{hidden_wide, Tensor{}};
        RankTensors rank_out{wide_out, Tensor{}};
        run_mtp({&m, nullptr}, rank_ids, embeddings != nullptr ? &rank_embeddings : nullptr,
                rank_hidden, rank_out, append_only);
    }

private:
    [[nodiscard]] static std::size_t at(int r) { return static_cast<std::size_t>(r); }
    [[nodiscard]] const Parameters& parameters(int r) const { return *r_[at(r)].parameters; }
    [[nodiscard]] const Qwen4Parameters& qwen4(int r) const { return *parameters(r).qwen4; }
    // Text block `layer`, or the MTP layer's block at layer == mtp_layer() (run_mtp).
    [[nodiscard]] const Qwen4BlockParameters& block(int r, std::size_t layer) const {
        if (layer == mtp_layer()) {
            const Qwen4MtpParameters* m = mtp_[at(r)];
            if (m == nullptr) { throw std::logic_error("Qwen4Exp MTP block outside run_mtp"); }
            return m->layer;
        }
        return qwen4(r).layers[layer];
    }
    // The MTP layer's index: after the Text layers (the routed-expert backends' layer number).
    [[nodiscard]] std::size_t mtp_layer() const { return config_.layer_types.size(); }

    [[nodiscard]] const char* phase() const {
        const Qwen4Columns& col = r_[0].columns;
        return col.prefill ? "prefill" : (col.replay_records != nullptr ? "verify" : "decode");
    }

    // The residual buffers: the caller's [S*H, T] outputs when executing; planning allocates them
    // in the caller's scope, so only execution needs the tensors.
    RankTensors wide_buffers(RankTensors& wide_out, std::int32_t T) {
        RankTensors out;
        if constexpr (kExecute) {
            for (int r = 0; r < width_; ++r) {
                Tensor& w = wide_out[at(r)];
                if (w.dtype != DType::BF16 || w.ne[0] != wide_rows() ||
                    w.numel() != static_cast<std::int64_t>(wide_rows()) * T || !w.is_contiguous() ||
                    w.data == nullptr) {
                    throw std::invalid_argument(
                        "Qwen4Exp wide residual output must be BF16 [S*H,T] on every rank");
                }
                out[at(r)] = w.view({wide_rows(), T});
            }
        } else {
            (void)wide_out;
            (void)T;
        }
        return out;
    }
    [[nodiscard]] Arena& work(int r) const { return *r_[at(r)].work; }
    [[nodiscard]] cudaStream_t s(int r) const { return r_[at(r)].stream; }
    [[nodiscard]] const Qwen4Columns& c(int r) const { return r_[at(r)].columns; }
    [[nodiscard]] const Qwen4State& state(int r) const { return r_[at(r)].state; }
    // The per-rank geometry (QSA and GDN heads); identical on both ranks.
    [[nodiscard]] const TextConfig& rc() const { return *r_[0].config; }

    [[nodiscard]] std::int32_t columns() const {
        if constexpr (kExecute) {
            return r_[0].columns.width * r_[0].columns.batch;
        } else {
            return last_;
        }
    }

    // Runs body(r) for each rank, with that rank's device current at tp 2 when executing. The
    // calls only enqueue, so the ranks' work overlaps on the devices.
    template <class Body>
    void each(Body&& body) {
        if constexpr (kExecute) {
            if (width_ == 2) {
                const ScopedCurrentDevice restore;
                for (int r = 0; r < width_; ++r) {
                    ScopedCurrentDevice::select(r_[at(r)].device);
                    body(r);
                }
                return;
            }
        }
        for (int r = 0; r < width_; ++r) { body(r); }
    }

    struct Scopes {
        std::array<std::optional<typename Arena::Scope>, 2> held;
    };

    [[nodiscard]] Scopes scopes() {
        Scopes out;
        for (int r = 0; r < width_; ++r) { out.held[at(r)].emplace(work(r).scope()); }
        return out;
    }

    [[nodiscard]] RankTensors alloc(DType dtype, std::initializer_list<std::int32_t> shape) {
        RankTensors out;
        for (int r = 0; r < width_; ++r) { out[at(r)] = work(r).alloc(dtype, shape); }
        return out;
    }

    // Planning: scratch of `bytes` on `rank` (every rank when negative).
    void reserve(std::size_t bytes, int rank = -1) {
        if constexpr (!kExecute) {
            if (bytes == 0) { return; }
            for (int r = 0; r < width_; ++r) {
                if (rank >= 0 && rank != r) { continue; }
                auto scope = work(r).scope();
                (void)work(r).alloc_bytes(bytes);
            }
        }
    }

    void linear(int r, const Tensor& x, const LinearParameters& p, Tensor& out) {
        if constexpr (kExecute) {
            ops::linear(x, p.weight, out, p.policy, work(r), s(r));
        } else {
            reserve(ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                         p.policy, first_, last_),
                    r);
        }
    }

    // A mixer's output projection: at one device a Linear; at tp 2 each rank contracts its head
    // block through its column shard and one BF16 all-reduce leaves the identical sum on both.
    // T128: the taps of the exchanges, when a teacher-forcing dump asks for them (tp 2 prefill).
    [[nodiscard]] Qwen4Taps* exchange_taps() const {
        if constexpr (kExecute) {
            const Qwen4Runtime* runtime = state(0).runtime;
            if (width_ == 2 && c(0).prefill && runtime != nullptr && runtime->taps != nullptr &&
                runtime->taps->wants_exchange()) {
                return runtime->taps;
            }
        }
        return nullptr;
    }

    // T128c: the mixer exchange carries FP32 partials (default; NINFER_TP_MIX=bf16 = BF16 partials) (FP16 payload in prefill,
    // FP32 in decode) and rounds to BF16 once, as at one device. Read once.
    static bool mix_fp32_partials() {
        static const bool on = [] {
            // Default since T138 (14/10): on (no measurable decode/prefill cost; E-fc02 step dispersion
            // 7.9 = GB300 8.1 vs 13.2 with the BF16 mixer exchange). NINFER_TP_MIX=bf16 restores BF16.
            const char* v = std::getenv("NINFER_TP_MIX");
            const bool enabled = v == nullptr || std::string(v) != "bf16";
            if (enabled) { std::fprintf(stderr, "T128c mixer exchange: FP32 partials, FP16/FP32 transport\n"); }
            return enabled;
        }();
        return on;
    }

    void output_projection(const RankTensors& x, std::array<const LinearParameters*, 2> p,
                           RankTensors& y, const RankTensors& staging, std::int32_t layer = -1) {
        if (width_ == 1) {
            linear(0, x[0], *p[0], y[0]);
            return;
        }
        if (mix_fp32_partials() && p[0]->weight.qtype == QType::Q8_G32_FP16) {
            const std::int32_t T = columns();
            auto scope           = scopes();
            RankTensors part     = alloc(DType::FP32, {hidden(), T});
            RankTensors stage    = alloc(DType::FP32, {hidden(), T});
            each([&](int r) { linear(r, x[at(r)], *p[at(r)], part[at(r)]); });
            if constexpr (kExecute) {
                (void)staging;
                if (Qwen4Taps* taps = exchange_taps()) {   // T128c check: FP32 partials of the mixer
                    taps->capture_exchange(layer, 0, part[0], part[1], c(0).first_position, s(0), s(1));
                }
                ops::HostExchange* pipe = c(0).prefill ? tp_.events->host_exchange() : nullptr;
                if (pipe != nullptr && pipe->begin_sum_f32(part, stage, &y, ops::HostExchange::Payload::FP16,
                                                           *tp_.execution)) {
                    pipe->columns_ready(0, 0, T);
                    pipe->columns_ready(1, 0, T);
                    pipe->finish_sum_f32();
                } else {
                    ops::allreduce_sum_f32(part, stage, &y, *tp_.execution, *tp_.events);
                }
            }
            return;
        }
        if constexpr (kExecute) {
            if (Qwen4Taps* taps = exchange_taps()) {
                // linear_row_parallel unrolled: each rank's partial, the tap, the BF16 all-reduce.
                each([&](int r) { linear(r, x[at(r)], *p[at(r)], y[at(r)]); });
                taps->capture_exchange(layer, 0, y[0], y[1], c(0).first_position, s(0), s(1));
                ops::allreduce_sum(y, staging, *tp_.execution, *tp_.events);
                return;
            }
            const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
            project_row_parallel(x, p, y, staging, ws, *tp_.execution, *tp_.events);
        } else {
            for (int r = 0; r < width_; ++r) {
                const LinearParameters& rp = *p[at(r)];
                reserve(ops::linear_workspace_capacity_bytes(rp.weight.qtype, rp.weight.n,
                                                             rp.weight.k, rp.policy, first_, last_),
                        r);
            }
        }
    }

    // Copy of rows [row0, row0+rows) into `destination` [rows, T] (always a copy).
    void copy_rows(int r, const Tensor& source, std::int32_t row0, std::int32_t rows,
                   const Tensor& destination) {
        if constexpr (kExecute) {
            const std::size_t element  = dtype_size(source.dtype);
            const std::int64_t columns = source.numel() / source.ne[0];
            // T139: a kernel, not a memcpy node: cudaGraphExecUpdate sometimes rejected the 2-D memcpy.
            ops::copy_rows(destination.data,
                           static_cast<const std::byte*>(source.data) +
                               static_cast<std::size_t>(row0) * element,
                           static_cast<std::size_t>(source.ne[0]) * element,
                           static_cast<std::size_t>(rows) * element,
                           static_cast<std::size_t>(columns), s(r));
        }
    }

    // Rows [row0, row0+rows) of a contiguous [R,T] matrix into `destination` [rows,T]; one column
    // is contiguous already and is returned as a view.
    Tensor split_rows(int r, const Tensor& source, std::int32_t row0, std::int32_t rows,
                      const Tensor& destination) {
        if constexpr (kExecute) {
            const std::size_t element  = dtype_size(source.dtype);
            const std::int64_t columns = source.numel() / source.ne[0];
            if (columns == 1) {
                Tensor view = destination;
                view.data   = static_cast<std::byte*>(source.data) +
                            static_cast<std::size_t>(row0) * element;
                return view;
            }
            // T139: a kernel, not a memcpy node: cudaGraphExecUpdate sometimes rejected the 2-D memcpy.
            ops::copy_rows(destination.data,
                           static_cast<const std::byte*>(source.data) +
                               static_cast<std::size_t>(row0) * element,
                           static_cast<std::size_t>(source.ne[0]) * element,
                           static_cast<std::size_t>(rows) * element,
                           static_cast<std::size_t>(columns), s(r));
        }
        return destination;
    }

    // ---- PLE (block 1, before its attention mixer), replicated on every rank -----------------
    void ple(std::size_t layer, const RankTensors& embedding, RankTensors& wide) {
        const std::int32_t T  = columns();
        const auto& ple       = *config_.ple;
        const std::int32_t S  = streams();
        auto scope            = scopes();
        RankTensors key_value = alloc(DType::BF16, {wide_rows() + hidden(), T});
        RankTensors gated     = alloc(DType::BF16, {wide_rows(), T});
        RankTensors normed    = alloc(DType::BF16, {wide_rows(), T});
        RankTensors slots     = alloc(DType::I32, {2});
        each([&](int r) {
            const PleParameters& p = *block(r, layer).ple;
            linear(r, embedding[at(r)], p.key_value, key_value[at(r)]);
            if constexpr (kExecute) {
                const Qwen4Columns& col = c(r);
                const auto W = col.width, B = col.batch;
                ops::ple_gate(wide[at(r)], key_value[at(r)], p.query_norm, p.key_norm, S,
                              config_.rms_norm_eps, gated[at(r)], s(r));
                ops::grouped_offset_rmsnorm(gated[at(r)], p.conv_norm, S, config_.rms_norm_eps,
                                            normed[at(r)], s(r));
                Tensor states = state(r).runtime->ple_states;
                Tensor source, destination;
                if (col.prefill) {
                    source      = slots[at(r)].slice(0, 0, 1);
                    destination = slots[at(r)].slice(0, 1, 1);
                    ops::set_i32_scalar(source, col.source_slot, s(r));
                    ops::set_i32_scalar(destination, col.destination_slot, s(r));
                } else {
                    source      = *col.source_slots;
                    destination = *col.destination_slots;
                }
                Tensor normed3 = normed[at(r)].view({wide_rows(), W, B});
                Tensor gated3  = gated[at(r)].view({wide_rows(), W, B});
                Tensor wide3   = wide[at(r)].view({wide_rows(), W, B});
                ops::ple_dilated_conv(normed3, gated3, p.convolution,
                                      static_cast<std::int32_t>(ple.conv_dilation()), states,
                                      source, wide3, s(r));
                if (col.replay_records != nullptr) {
                    // Verification: the history advances at commit by the accepted prefix.
                    const Tensor& record = state(r).runtime->ple_record;
                    if (record.data == nullptr || record.ne[0] != wide_rows() ||
                        record.ne[1] != W || record.ne[2] < B) {
                        throw std::logic_error("Qwen4Exp PLE speculative record is unavailable");
                    }
                    CUDA_CHECK(::ninfer::memcpy_device_async(record.data, normed[at(r)].data,
                                               normed[at(r)].bytes(),
                                               s(r)));
                } else {
                    ops::ple_conv_advance(normed3, col.prefill ? nullptr : col.valid_columns,
                                          static_cast<std::int32_t>(ple.conv_history()), states,
                                          source, destination, s(r));
                }
            }
        });
    }

    // ---- QSA: gated attention over the indexer's selection -----------------------------------
    // Each rank: its query/KV heads, the replicated indexer (identical selection on both ranks).
    // `y` null: append K/V and the index records only (MTP prompt columns).
    void qsa(std::size_t layer, const RankTensors& x, RankTensors* y, std::uint32_t kv_layer,
             const RankTensors& staging) {
        const std::int32_t T  = columns();
        const auto& a         = *rc().attention;
        const auto head_dim   = dimension(a.head_dim);
        const auto heads      = dimension(a.num_attention_heads);
        const auto kv_heads   = dimension(a.num_key_value_heads);
        const auto qw         = dimension(a.query_width());
        const auto kw         = dimension(a.key_width());
        const auto geometry   = qwen4_qsa_geometry(rc());
        const auto index_rows = geometry.index_heads * geometry.index_dim;
        auto scope            = scopes();
        RankTensors projection  = alloc(DType::BF16, {2 * qw + 2 * kw, T});
        RankTensors q           = alloc(DType::BF16, {qw, T});
        RankTensors k           = alloc(DType::BF16, {kw, T});
        RankTensors gate        = alloc(DType::BF16, {qw, T});
        RankTensors v           = alloc(DType::BF16, {kw, T});
        RankTensors qn          = alloc(DType::BF16, {head_dim, heads, T});
        RankTensors kn          = alloc(DType::BF16, {head_dim, kv_heads, T});
        RankTensors indexer     = alloc(DType::BF16, {index_rows + geometry.index_dim, T});
        RankTensors index_query = alloc(DType::BF16, {index_rows, T});
        RankTensors index_key   = alloc(DType::BF16, {geometry.index_dim, T});
        const std::int32_t W = c(0).width, B = c(0).batch;
        RankTensors selected = alloc(DType::I32, {geometry.max_selected(), kExecute ? W : T,
                                                  kExecute ? B : 1});
        RankTensors counts   = alloc(DType::I32, {kExecute ? W : T, kExecute ? B : 1});
        RankTensors attended = alloc(DType::BF16, {head_dim, heads, T});
        const auto p = [&](int r) -> const QsaParameters& {
            return std::get<QsaParameters>(block(r, layer).mixer);
        };
        each([&](int r) {
            linear(r, x[at(r)], p(r).projection, projection[at(r)]);
            linear(r, x[at(r)], p(r).indexer, indexer[at(r)]);
        });
        if constexpr (!kExecute) {
            // Selection scratch for the planned window: W columns of B lanes.
            reserve(ops::qsa_select_workspace_bytes(geometry, std::max(c(0).max_visible, 1U), W, B));
            reserve(ops::qsa_attention_workspace_bytes(geometry, last_));
        } else {
            each([&](int r) {
                const std::size_t i     = at(r);
                const QsaParameters& qp = p(r);
                const Qwen4Columns& col = c(r);
                cudaStream_t stream     = s(r);
                const Tensor qs = split_rows(r, projection[i], 0, qw, q[i]);
                const Tensor ks = split_rows(r, projection[i], qw, kw, k[i]);
                const Tensor gs = split_rows(r, projection[i], qw + kw, qw, gate[i]);
                const Tensor vs = split_rows(r, projection[i], 2 * qw + kw, kw, v[i]);
                const Tensor iq = split_rows(r, indexer[i], 0, index_rows, index_query[i]);
                const Tensor ik =
                    split_rows(r, indexer[i], index_rows, geometry.index_dim, index_key[i]);
                if (y != nullptr) {
                    ops::rmsnorm(qs.view({head_dim, heads, T}), qp.query_norm,
                                 config_.rms_norm_eps, true, qn[i], stream);
                }
                ops::rmsnorm(ks.view({head_dim, kv_heads, T}), qp.key_norm, config_.rms_norm_eps,
                             true, kn[i], stream);
                const Tensor& rope = *col.rope_positions;
                // Only prefill binds axis-major [T,3] MRoPE rows; decode windows bind [W,B]
                // positions.
                const bool mrope       = col.prefill && rope.ne[1] == 3;
                const Tensor rope_flat = mrope ? rope : rope.view({T});
                if (y != nullptr) {
                    text_rope(rope_flat, *config_.rope_parameters, qn[i], kn[i], stream);
                } else {
                    text_rope(rope_flat, *config_.rope_parameters, kn[i], stream);
                }
                const Tensor rope_rows = mrope ? rope : rope.view({T, 1});
                const Tensor positions = col.cache_positions->view({W, B});
                const Tensor valid = col.valid_columns != nullptr ? *col.valid_columns : Tensor{};
                const PagedKVBatchLayerView kv  = state(r).kv->batch_layer_view(kv_layer);
                const ops::QsaIndexPlanes index = state(r).kv->qsa_index_planes(kv_layer);
                ops::qsa_append(kn[i].view({head_dim, kv_heads, W, B}),
                                vs.view({head_dim, kv_heads, W, B}),
                                ik.view({geometry.index_dim, W, B}), rope_rows, positions, valid,
                                *col.table_rows, qp.indexer_key_norm, geometry, kv, index, stream);
                if (y == nullptr) { return; }
                ops::qsa_select(iq.view({index_rows, W, B}), rope_rows, positions, valid,
                                *col.table_rows, qp.indexer_query_norm, index, geometry,
                                col.max_visible, work(r), selected[i], counts[i], stream);
                Tensor a4 = attended[i].view({head_dim, heads, W, B});
                ops::qsa_attention(qn[i].view({head_dim, heads, W, B}), selected[i], counts[i],
                                   *col.table_rows, kv,
                                   static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))),
                                   work(r), a4, stream);
                Tensor flat = attended[i].view({qw, T});
                ops::sigmoid_mul(gs, flat, stream);
            });
        }
        if (y == nullptr) { return; }
        const RankTensors flat{attended[0].view({qw, T}),
                               width_ == 2 ? attended[1].view({qw, T}) : Tensor{}};
        output_projection(flat, {&p(0).output, width_ == 2 ? &p(1).output : nullptr}, *y, staging,
                          static_cast<std::int32_t>(layer));
    }

    // ---- GDN: each rank's key/value heads, conv channels and state slots ---------------------
    void gdn(std::size_t layer, const RankTensors& x, RankTensors& y, std::int32_t gidx,
             const RankTensors& staging) {
        const std::int32_t T = columns();
        const auto& g        = *rc().gdn;
        const auto heads     = dimension(g.linear_num_value_heads);
        const auto kheads    = dimension(g.linear_num_key_heads);
        const auto dk        = dimension(g.linear_key_head_dim);
        const auto dv        = dimension(g.linear_value_head_dim);
        const auto kw        = dimension(g.key_width());
        const auto vw        = dimension(g.value_width());
        const auto conv      = dimension(g.conv_channels());
        auto scope              = scopes();
        RankTensors projection  = alloc(DType::BF16, {conv + vw, T});
        RankTensors control     = alloc(DType::BF16, {2 * heads, T});
        RankTensors decay       = alloc(DType::FP32, {heads, T});
        RankTensors beta        = alloc(DType::FP32, {heads, T});
        RankTensors qkv         = alloc(DType::BF16, {conv, T});
        RankTensors z           = alloc(DType::BF16, {vw, T});
        RankTensors qc          = alloc(DType::BF16, {kw, T});
        RankTensors kc          = alloc(DType::BF16, {kw, T});
        RankTensors vc          = alloc(DType::BF16, {vw, T});
        RankTensors o           = alloc(DType::BF16, {dv, heads, T});
        RankTensors on          = alloc(DType::BF16, {dv, heads, T});
        const auto p = [&](int r) -> const Qwen4GdnParameters& {
            return std::get<Qwen4GdnParameters>(block(r, layer).mixer);
        };
        each([&](int r) {
            linear(r, x[at(r)], p(r).projection, projection[at(r)]);
            linear(r, x[at(r)], p(r).control, control[at(r)]);
        });
        if constexpr (!kExecute) {
            if (c(0).prefill) {
                reserve(ops::gated_delta_net_workspace_capacity_bytes(kheads, heads, first_,
                                                                      last_));
            }
        } else {
            each([&](int r) {
                const std::size_t i          = at(r);
                const Qwen4GdnParameters& gp = p(r);
                const Qwen4Columns& col      = c(r);
                LinearAttentionStatePool& pool = *state(r).linear;
                cudaStream_t stream          = s(r);
                // a|b read in place from the projection (T49: no row-split copies).
                const Tensor as = control[i].slice(0, 0, heads);
                const Tensor bs = control[i].slice(0, heads, heads);
                ops::gdn_gating(as, bs, gp.a_log, gp.dt_bias, decay[i], beta[i], stream);
                const Tensor zs   = split_rows(r, projection[i], conv, vw, z[i]);
                const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dk)));
                const auto gl     = static_cast<std::uint32_t>(gidx);
                if (col.prefill) {
                    const Tensor qkvs = split_rows(r, projection[i], 0, conv, qkv[i]);
                    Tensor conv_in  = pool.conv_slot(gl, col.source_slot);
                    Tensor conv_out = pool.conv_slot(gl, col.destination_slot);
                    ops::causal_conv1d_silu_split(qkvs, gp.convolution, conv_in, conv_out, qc[i],
                                                  kc[i], vc[i], stream);
                    Tensor in  = pool.recurrent_slot(gl, col.source_slot);
                    Tensor out = pool.recurrent_slot(gl, col.destination_slot);
                    ops::gated_delta_net(qc[i].view({dk, kheads, T}), kc[i].view({dk, kheads, T}),
                                         vc[i].view({dv, heads, T}), decay[i], beta[i], scale,
                                         /*normalize_qk=*/true, work(r), in, out, o[i],
                                         r_[i].execution);
                } else {
                    const std::int32_t W = col.width, B = col.batch;
                    Tensor conv_states   = pool.layer_view(gl).conv;
                    const Tensor valid =
                        col.valid_columns != nullptr ? *col.valid_columns : Tensor{};
                    Tensor q3 = qc[i].view({kw, W, B});
                    Tensor k3 = kc[i].view({kw, W, B});
                    Tensor v3 = vc[i].view({vw, W, B});
                    Tensor recurrent = pool.layer_view(gl).recurrent;
                    Tensor o4        = o[i].view({dv, heads, W, B});
                    if (col.replay_records != nullptr) {
                        // ReplaySSM verification: the projected conv inputs are the record itself.
                        GdnReplayRecordLayer records = col.replay_records->layer(gidx, B);
                        if (records.conv.ne[1] != W) {
                            throw std::logic_error(
                                "Qwen4Exp ReplaySSM record width is not the window");
                        }
                        copy_rows(r, projection[i], 0, conv, records.conv.view({conv, T}));
                        ops::gdn_projected_conv_record(records.conv, gp.convolution, conv_states,
                                                       valid, *col.source_slots, q3, k3, v3,
                                                       stream);
                        ops::gated_delta_net_replay_record(
                            qc[i].view({dk, kheads, W, B}), kc[i].view({dk, kheads, W, B}),
                            vc[i].view({dv, heads, W, B}), decay[i].view({heads, W, B}),
                            beta[i].view({heads, W, B}), scale, recurrent, valid,
                            *col.source_slots, records.key, records.value, records.gate, o4,
                            stream);
                    } else {
                        const Tensor qkvs = split_rows(r, projection[i], 0, conv, qkv[i]);
                        ops::gdn_projected_conv_snapshot(qkvs.view({conv, W, B}), gp.convolution,
                                                         conv_states, valid, *col.source_slots,
                                                         *col.destination_slots, q3, k3, v3,
                                                         stream);
                        ops::gated_delta_net_batch_update(
                            qc[i].view({dk, kheads, W, B}), kc[i].view({dk, kheads, W, B}),
                            vc[i].view({dv, heads, W, B}), decay[i].view({heads, W, B}),
                            beta[i].view({heads, W, B}), scale, /*normalize_qk=*/true, recurrent,
                            *col.source_slots, *col.destination_slots, o4, stream);
                    }
                }
                ops::gated_rmsnorm_sigmoid(o[i], gp.norm, zs.view({dv, heads, T}),
                                           config_.rms_norm_eps, on[i], stream);
            });
        }
        const RankTensors flat{on[0].view({vw, T}), width_ == 2 ? on[1].view({vw, T}) : Tensor{}};
        output_projection(flat, {&p(0).output, width_ == 2 ? &p(1).output : nullptr}, y, staging,
                          static_cast<std::int32_t>(layer));
    }

    // T25 router-ahead prefetch: `wide` holds layer's committed residual (its MLP
    // hyper-connection input); the backend may predict layer+1's experts from it. Decode rounds at
    // one device only (the offload backend's prefetch); a no-op for the other backends.
    void ahead(std::size_t layer, const RankTensors& wide) {
        if constexpr (kExecute) {
            if (width_ != 1 || r_[0].columns.prefill || layer + 1 >= config_.layer_types.size()) {
                return;
            }
            const Qwen4BlockParameters& next = block(0, layer + 1);
            routed_experts(0, layer)->ahead(static_cast<std::int32_t>(layer), wide[0],
                                            next.ffn_hc.weights.norm, next.moe.router,
                                            config_.rms_norm_eps, s(0));
        } else {
            (void)layer, (void)wide;
        }
    }

    // ---- MoE: router, shared expert, routed experts (FP32 partial), one rounding --------------
    // At tp 2 each rank routes the same tokens (replicated router), runs the routed experts of its
    // partition into its FP32 partial, the shared expert's owner adds the gated shared expert in
    // FP32, and one FP32 all-reduce rounds the block output to BF16 once on both ranks.
    // `layer` is a Text layer or mtp_layer() (the MTP layer's backend and bank).
    void moe(std::size_t layer, const RankTensors& x, RankTensors& y, const RankTensors& staging) {
        const std::int32_t T  = columns();
        const auto& m         = std::get<MoeConfig>(config_.ffn);
        const auto shared_w   = dimension(m.shared_expert_intermediate_size);
        auto scope            = scopes();
        RankTensors ids        = alloc(DType::I32, {ops::kMoeRouteTopK, T});
        RankTensors weights    = alloc(DType::FP32, {ops::kMoeRouteTopK, T});
        RankTensors gate       = alloc(DType::FP32, {T});
        RankTensors gate_up    = alloc(DType::BF16, {2 * shared_w, T});
        RankTensors activation = alloc(DType::BF16, {shared_w, T});
        RankTensors shared     = alloc(DType::BF16, {hidden(), T});
        RankTensors partial    = alloc(DType::FP32, {hidden(), T});
        reserve(ops::moe_route_workspace_bytes(T));
        const auto layer_index = static_cast<std::int32_t>(layer);
        each([&](int r) {
            const std::size_t i         = at(r);
            const Qwen4MoeParameters& p = block(r, layer).moe;
            if constexpr (kExecute) {
                // Verify / MTP columns past a lane's valid count route nothing (never committed:
                // no CPU expert work, no cache-policy accesses). The MTP alignment routes only
                // the accepted column (T52): the other columns' outputs are never read.
                const Qwen4Columns& col = c(r);
                const Tensor* valid     = col.prefill ? nullptr : col.valid_columns;
                ops::moe_route(x[i], p.router, work(r), ids[i], weights[i], gate[i], s(r), valid,
                               valid != nullptr && col.moe_last_valid_only
                                   ? ops::MoeRouteColumns::LastValid
                                   : ops::MoeRouteColumns::ValidPrefix);
                // The backend may start the routed experts elsewhere (the CPU) right away; what
                // it allocates lives until this scope ends.
                routed_experts(r, layer)->routed(layer_index, expert_bank(r, layer), x[i], ids[i],
                                                 weights[i], work(r), s(r));
            } else {
                const std::size_t routed_bytes = Qwen4RoutedExperts::routed_workspace_bytes(T);
                if (routed_bytes != 0) { (void)work(r).alloc_bytes(routed_bytes); }
            }
            if (p.shared) {
                linear(r, x[i], p.shared->gate_up, gate_up[i]);
                if constexpr (kExecute) {
                    ops::silu_mul(gate_up[i].slice(0, 0, shared_w),
                                  gate_up[i].slice(0, shared_w, shared_w), activation[i], s(r));
                }
                linear(r, activation[i], p.shared->down, shared[i]);
            }
        });
        reserve(Qwen4RoutedExperts::workspace_bytes(T));
        if constexpr (kExecute) {
            // T62: at tp 2 a prompt chunk's MoE exchange may take the pipelined host exchange
            // (optionally a narrow payload, optionally by column groups that overlap the routed
            // experts of the later columns). Decode rounds (prefill == false) never do.
            const T62Exchange& t62 = t62_exchange();
            ops::HostExchange* pipe =
                width_ == 2 && c(0).prefill && t62.enabled ? tp_.events->host_exchange() : nullptr;
            if (pipe != nullptr &&
                !pipe->begin_sum_f32(partial, staging, &y, t62.payload, *tp_.execution)) {
                pipe = nullptr;
            }
            each([&](int r) {
                const std::size_t i         = at(r);
                const Qwen4MoeParameters& p = block(r, layer).moe;
                const Qwen4Runtime& runtime = *state(r).runtime;
                if (runtime.moe_probe != nullptr && runtime.moe_probe->wants(layer_index)) {
                    runtime.moe_probe->capture(layer_index, x[i], ids[i], weights[i], s(r));
                }
                // Columns [first, first + n) of this rank's partial are final: add the shared
                // expert there (owner rank; elementwise, the same bits as on the whole) and send.
                const auto ready = [&, r, i](std::int32_t first, std::int32_t n) {
                    if (p.shared) {
                        Tensor part = partial[i].slice(1, first, n);
                        ops::moe_accumulate_shared(gate[i].slice(0, first, n),
                                                   shared[i].slice(1, first, n), part, s(r));
                    }
                    pipe->columns_ready(r, first, n);
                };
                Qwen4ColumnGroups groups;
                if (pipe != nullptr && t62.group_columns > 0) {
                    groups.group_columns = t62.group_columns;
                    groups.ready         = ready;
                    qwen4_column_groups  = &groups;
                }
                routed_experts(r, layer)->run(layer_index, expert_bank(r, layer), x[i], ids[i],
                                              weights[i], partial[i], work(r), s(r));
                qwen4_column_groups = nullptr;
                if (width_ == 1) {
                    ops::moe_combine_shared(partial[i], gate[i], shared[i], y[i], s(r));
                } else if (pipe != nullptr) {
                    if (groups.done < T) { ready(groups.done, T - groups.done); }
                } else if (p.shared) {
                    ops::moe_accumulate_shared(gate[i], shared[i], partial[i], s(r));
                }
            });
            if (width_ == 2) {
                if (pipe != nullptr) {
                    pipe->finish_sum_f32();
                } else {
                    if (Qwen4Taps* taps = exchange_taps()) {
                        taps->capture_exchange(layer_index, 1, partial[0], partial[1],
                                               c(0).first_position, s(0), s(1));
                    }
                    ops::allreduce_sum_f32(partial, staging, &y, *tp_.execution, *tp_.events);
                }
                each([&](int r) { routed_experts(r, layer)->exchanged(layer_index, s(r)); });
            }
        }
    }

    // T62 switches, read once. Default since T127: FP16 payload (T62 adopted BF16), column
    // groups of 2048. NINFER_TP_AR=0 restores the T56 exchange (allreduce_sum_f32, FP32, one
    // call); NINFER_TP_AR_PAYLOAD = fp32 | fp16 | bf16 (fp16 measured, rejected on quality);
    // NINFER_TP_AR_GROUP = column group of the overlap (rounded down to whole 2048-column
    // pieces; 0 = one group); NINFER_TP_AR_PIPE = 1 takes the pipelined exchange even at fp32
    // with one group (control arm).
    struct T62Exchange {
        bool enabled               = false;
        // T127 (13/10): default FP16 (same bytes and speed as BF16, 8x smaller rounding of the partial sums;
        // T128: error added vs one device 1.44x (BF16) -> 1.01x (FP16) of the unavoidable BF16 rounding). NINFER_TP_AR_PAYLOAD overrides.
        ops::HostExchange::Payload payload = ops::HostExchange::Payload::FP16;
        std::int32_t group_columns = ops::kMoeGroupedChunkColumns;
    };
    static const T62Exchange& t62_exchange() {
        static const T62Exchange value = [] {
            T62Exchange v;
            const auto get = [](const char* name) -> std::string {
                const char* e = std::getenv(name);
                return e != nullptr ? std::string(e) : std::string();
            };
            if (get("NINFER_TP_AR") == "0") { return v; } // enabled == false: T56 path
            const std::string payload = get("NINFER_TP_AR_PAYLOAD");
            if (payload == "fp32") { v.payload = ops::HostExchange::Payload::FP32; }
            if (payload == "fp16") { v.payload = ops::HostExchange::Payload::FP16; }
            if (payload == "bf16") { v.payload = ops::HostExchange::Payload::BF16; }
            const std::string group = get("NINFER_TP_AR_GROUP");
            if (!group.empty()) {
                const auto g    = static_cast<std::int32_t>(std::strtol(group.c_str(), nullptr, 10));
                const auto unit = ops::kMoeGroupedChunkColumns;
                v.group_columns = g > 0 ? std::max(unit, g / unit * unit) : 0;
            }
            v.enabled = v.payload != ops::HostExchange::Payload::FP32 || v.group_columns > 0 ||
                        get("NINFER_TP_AR_PIPE") == "1";
            if (v.enabled) {
                const char* names[] = {"fp32", "fp16", "bf16"};
                std::fprintf(stderr, "T62 prefill exchange: payload %s, group %d columns\n",
                             names[static_cast<int>(v.payload)], v.group_columns);
            }
            return v;
        }();
        return value;
    }

    // The routed-expert backend and bank of `layer` on rank r (execution only).
    [[nodiscard]] Qwen4RoutedExperts* routed_experts(int r, std::size_t layer) const {
        const Qwen4Runtime* runtime = state(r).runtime;
        Qwen4RoutedExperts* experts =
            runtime == nullptr ? nullptr
                               : (layer == mtp_layer() ? runtime->mtp_experts : runtime->experts);
        if (experts == nullptr) {
            throw std::logic_error(layer == mtp_layer()
                                       ? "Qwen4Exp MTP layer has no bound expert backend"
                                       : "Qwen4Exp MoE has no routed-expert backend");
        }
        return experts;
    }
    [[nodiscard]] const Qwen4ExpertBank& expert_bank(int r, std::size_t layer) const {
        if (layer != mtp_layer()) { return block(r, layer).moe.experts; }
        const Qwen4Runtime* runtime = state(r).runtime;
        if (runtime == nullptr || runtime->mtp_expert_bank == nullptr) {
            throw std::logic_error("Qwen4Exp MTP layer has no bound expert bank");
        }
        return *runtime->mtp_expert_bank;
    }

    const TextConfig& config_;
    std::array<Rank, 2> r_;
    std::array<const Qwen4MtpParameters*, 2> mtp_{};
    int width_ = 1;
    Qwen4Collectives tp_;
    std::int32_t first_ = 1;
    std::int32_t last_  = 1;
};

} // namespace ninfer::models::qwen3_5::execution
