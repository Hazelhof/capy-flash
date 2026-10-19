#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t
attention_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                     std::int32_t last);
void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream);

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream);
void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               cudaStream_t stream);

// rmsnorm (unit offset) of query and key into query_out and key_out, then text_rope of the pair.
// The two-device 27B attention shard ([256,12,T] / [256,2,T], R=64) runs it as one fused Op with
// the same arithmetic; other geometries run the three Ops.
void text_qk_norm_rope(const Tensor& positions, const RopeConfig& config, const Tensor& query_norm,
                       const Tensor& key_norm, float eps, const Tensor& query, const Tensor& key,
                       Tensor& query_out, Tensor& key_out, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
