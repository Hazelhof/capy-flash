#pragma once

// ninfer::ops::detail - private launch prototype for sample.

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream);

// replace_banned_token: masks the banned ids when out[0] is one of them (flag = 1) and copies the
// configuration without its penalty-count pointer into config_copy; then commits alt over out.
void banned_token_check_launch(Tensor& logits, const Tensor& out, std::int32_t banned0,
                               std::int32_t banned1, const SamplingConfig* configs,
                               SamplingConfig* config_copy, std::int32_t* flag,
                               cudaStream_t stream);
void banned_token_commit_launch(Tensor& out, const std::int32_t* alt, const std::int32_t* flag,
                                const SamplingConfig* configs, unsigned long long* replacements,
                                cudaStream_t stream);

void increment_token_counts_launch(const Tensor& token_ids, Tensor& token_counts,
                                   cudaStream_t stream);

[[nodiscard]] std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain,
                                                         std::int32_t columns);

} // namespace ninfer::ops::detail
