#include "models/qwen3_5/execution/attention.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/tp2/rmsnorm_rope.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

} // namespace

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& weight = single->weight;
        return ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                             single->policy, first, last);
    }
    return 0;
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(hidden, pair->first, pair->second, query, gate, key, value, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::attn_input_proj(hidden, single.weight, query, gate, key, value, single.policy,
                             workspace, stream);
    }
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, stream);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, key, stream);
}

void text_qk_norm_rope(const Tensor& positions, const RopeConfig& config, const Tensor& query_norm,
                       const Tensor& key_norm, float eps, const Tensor& query, const Tensor& key,
                       Tensor& query_out, Tensor& key_out, cudaStream_t stream) {
    require_rope_axes(positions, config);
    const std::int32_t tokens = query.ne[2];
    const bool shard          = dimension(config.rotary_dim) == 64 && query.ne[0] == 256 &&
                       query.ne[1] == 12 && key.ne[0] == 256 && key.ne[1] == 2 &&
                       (positions.ne[1] == 1 || positions.ne[1] == 3) && tokens > 0 &&
                       query.is_contiguous() && key.is_contiguous() && query_out.is_contiguous() &&
                       key_out.is_contiguous();
    if (shard) {
        ops::rmsnorm_rope_text_shard(positions, 64, config.rope_theta, query_norm, key_norm, eps,
                                     query, key, query_out, key_out, stream);
        return;
    }
    ops::rmsnorm(query, query_norm, eps, true, query_out, stream);
    ops::rmsnorm(key, key_norm, eps, true, key_out, stream);
    text_rope(positions, config, query_out, key_out, stream);
}

} // namespace ninfer::models::qwen3_5::execution
