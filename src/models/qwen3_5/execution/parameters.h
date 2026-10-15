#pragma once

#include "models/qwen3_5/execution/ple_gather.h"
#include "models/qwen3_5/model.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/moe_slots.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
};

using FfnParameters = std::variant<DenseParameters, ops::SparseMoeWeights>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters> mixer;
    FfnParameters ffn;
    ops::SparseMoeHints projection_prefetch;
};

struct TextParameters {
    Weight token_embedding;
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
};

struct MtpProjectionParameters {
    LinearParameters packed;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses its existing complete-parent Attention projection.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    // Empty on a tensor-parallel rank that does not hold the selected head: a PrimaryOnly
    // optimized proposal head proposes on rank 0 alone. A vocabulary-split one is this rank's
    // row block on both ranks.
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head; // This rank's rows: all of them, or a vocabulary block when split.
    std::optional<Tensor> token_ids; // Row -> token ID map; rank 0 only at tp 2.
    std::uint32_t rows = 0;          // Rows of the whole head.

    // At tp 2 under MTP each rank holds half of the rows (load/sharding.h).
    [[nodiscard]] bool split() const noexcept {
        return static_cast<std::uint64_t>(head.weight.n) != rows;
    }
};

// ---- Qwen4Exp (Qwen3.8-Flash-Next) -------------------------------------------------------------
// Names follow ninfer-ext's execution (giveen/ninfer-ext @9c933f01, Apache-2.0) and T12's
// bindings (weights.h `Qwen4*Weights`).

// One hyper-connection mixer: the T10 Op operands (down = [low rank (+ S inject rows), S*H]).
struct HyperConnectionParameters {
    ops::HyperConnectionWeights weights;
    bool inject = false;
};

struct QsaParameters {
    LinearParameters projection; // query, key, gate, value rows
    Tensor query_norm, key_norm;
    LinearParameters output;
    LinearParameters indexer; // indexer query rows, then the raw key row block
    Tensor indexer_query_norm, indexer_key_norm;
};

struct Qwen4GdnParameters {
    LinearParameters projection; // query, key, value, z rows
    LinearParameters control;    // a, b rows
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

// The routed experts of one layer as stored: the host-mapped NVFP4 banks (device-addressable
// pinned memory, ShardAxis::HostMapped) in the moe_slots plane description, expert e of a plane at
// base + e * stride; the stacked weight divisors (gate of e at 2e, up at 2e+1; down at e), also
// host-mapped; and the layer's two activation divisors.
struct Qwen4ExpertBank {
    ops::MoeSlotPool bank; // slots = experts
    const float* gate_up_divisors = nullptr;
    const float* down_divisors    = nullptr;
    float gate_up_input_divisor   = 0.0F;
    float down_input_divisor      = 0.0F;
};

// The shared expert (gate|up rows, down) of one block.
struct Qwen4SharedExpertParameters {
    LinearParameters gate_up;
    LinearParameters down;
};

struct Qwen4MoeParameters {
    Tensor router; // BF16 [H, E+1]: the routed rows then the shared-expert score row
    // Whole on every block at one device; at tp 2 held by rank (block % 2) alone (the MTP block
    // by rank 0, load/sharding.h), so empty on the other rank.
    std::optional<Qwen4SharedExpertParameters> shared;
    Qwen4ExpertBank experts;
};

struct PleParameters {
    PleTableView table;
    PleHashConfig hash;
    LinearParameters key_value; // key rows [S*H], then value rows [H]
    Tensor key_norm, query_norm, conv_norm, convolution;
};

struct Qwen4BlockParameters {
    HyperConnectionParameters attention_hc, ffn_hc;
    std::variant<QsaParameters, Qwen4GdnParameters> mixer;
    Qwen4MoeParameters moe;
    std::optional<PleParameters> ple;
};

struct Qwen4Parameters {
    HyperConnectionParameters head;
    std::vector<Qwen4BlockParameters> layers;
};

// The MTP layer (docs/maintainer/qwen4-exp-model.md#mtp): stem over the wide residual and the
// shifted token, one QSA block with its own 512 routed experts, and its own output mixer in front
// of the Text output head.
//
// Its routed experts reach the W4A4 expert Op like the Text banks, with two differences the
// preparation resolves (T24):
//   * weight divisors: one per matrix (this tree's recipes) or one for the whole bank (ninfer-ext's
//     artifact); `uniform_*_divisor` holds the latter, which the Program expands into per-expert
//     planes (moe_slots indexes divisors by expert id);
//   * activation divisors: the checkpoint calibrates none for the MTP experts (the recipes store
//     them A16Only). A calibrated pair stored with the bank (AllowA4) is used as is; otherwise the
//     pair is derived from the Text layers' calibrated ranges: the widest range of all 48 layers
//     times kQwen4MtpActivationRangeMargin. The W4A4 error is flat while the range lies between
//     the largest activation and ~64x it (E4M3 block scales keep three mantissa bits over that
//     span); on the real model the MTP inputs reach 24.5 (x) and 826 (h) against Text ranges of
//     13.9 and 57.8, and margins 16-64 give the minimum error with no saturated block, 4 saturates
//     h blocks (T24, docs/capy-flash/development-log.md). 32 centres both inputs in the flat region.
inline constexpr float kQwen4MtpActivationRangeMargin = 32.0F;

struct Qwen4MtpParameters {
    Tensor embedding_norm; // [H]
    Tensor hidden_norm;    // [S*H]: one RMS over the whole wide residual
    LinearParameters embedding_projection;
    LinearParameters hidden_projection;
    HyperConnectionParameters head; // no inject
    Qwen4BlockParameters layer;     // QSA mixer, MoE over the MTP banks
    float uniform_gate_up_divisor = 0.0F; // > 0: the bank keeps one weight divisor per projection
    float uniform_down_divisor    = 0.0F;
    bool derived_activation_divisors = false;
};

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
//
// A tensor-parallel Model has one Parameters per rank. Its operands are that rank's shards,
// with head, intermediate and vocabulary extents divided by tp (load/sharding.h); Vision, draft
// and proposal parameters exist only on the rank that holds them.
class Parameters {
public:
    explicit Parameters(const Model& source) : Parameters(source, 0) {}

    Parameters(const Model& source, int device);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    const int device;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
    // Qwen4Exp Text (text.layers is empty for it; text.token_embedding / output_head are used).
    // At tp 2: this rank's shards (QSA 12 query + 1 KV heads, GDN 8 key + 24 value heads, the
    // vocabulary half of the output head), the replicated router, indexer, PLE and
    // hyper-connection mixers, the shared expert of its blocks, and the one host-mapped copy of
    // the routed experts that both ranks read.
    std::optional<Qwen4Parameters> qwen4;
    // Qwen4Exp MTP layer, when the artifact's MTP component is loaded.
    std::optional<Qwen4MtpParameters> qwen4_mtp;
};

} // namespace ninfer::models::qwen3_5::execution
