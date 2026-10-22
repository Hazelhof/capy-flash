// Implements: include/ninfer/ops/sampling.h
// Match: validated contiguous BF16/I32 tensors and a shared-layout workspace.
// Algorithm assumptions: launcher and kernels use sampler_multiblock_ok() from
// the same layout authority, so exactly one finite route owns each shape.
#include "ops/launcher/sampling.h"

#include "ops/common/math.h"
#include "ops/kernel/sampling.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {

__global__ void increment_token_counts_kernel(const std::int32_t* token_ids, std::int32_t count,
                                              std::int32_t* token_counts) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) { atomicAdd(&token_counts[token_ids[index]], 1); }
}

__global__ void banned_token_check_kernel(__nv_bfloat16* logits, const std::int32_t* out,
                                          std::int32_t banned0, std::int32_t banned1,
                                          const SamplingConfig* configs,
                                          SamplingConfig* config_copy, std::int32_t* flag) {
    if (threadIdx.x != 0) { return; }
    SamplingConfig copy = configs[0];
    // The second draw must not count itself: the commit moves the count when it replaces.
    copy.token_counts   = nullptr;
    config_copy[0]      = copy;
    const std::int32_t token = out[0];
    const bool hit           = token >= 0 && (token == banned0 || token == banned1);
    flag[0]                  = hit ? 1 : 0;
    if (!hit) { return; }
    const __nv_bfloat16 masked = __float2bfloat16(-CUDART_INF_F);
    if (banned0 >= 0) { logits[banned0] = masked; }
    if (banned1 >= 0) { logits[banned1] = masked; }
}

__global__ void banned_token_commit_kernel(std::int32_t* out, const std::int32_t* alt,
                                           const std::int32_t* flag,
                                           const SamplingConfig* configs,
                                           unsigned long long* replacements) {
    if (threadIdx.x != 0 || flag[0] == 0) { return; }
    const std::int32_t old_token = out[0];
    const std::int32_t new_token = alt[0];
    if (configs[0].token_counts != nullptr) {
        atomicSub(&configs[0].token_counts[old_token], 1);
        atomicAdd(&configs[0].token_counts[new_token], 1);
    }
    out[0] = new_token;
    if (replacements != nullptr) { atomicAdd(replacements, 1ULL); }
}

void banned_token_check_launch(Tensor& logits, const Tensor& out, std::int32_t banned0,
                               std::int32_t banned1, const SamplingConfig* configs,
                               SamplingConfig* config_copy, std::int32_t* flag,
                               cudaStream_t stream) {
    banned_token_check_kernel<<<1, 32, 0, stream>>>(
        static_cast<__nv_bfloat16*>(logits.data), static_cast<const std::int32_t*>(out.data),
        banned0, banned1, configs, config_copy, flag);
    CUDA_CHECK(cudaGetLastError());
}

void banned_token_commit_launch(Tensor& out, const std::int32_t* alt, const std::int32_t* flag,
                                const SamplingConfig* configs, unsigned long long* replacements,
                                cudaStream_t stream) {
    banned_token_commit_kernel<<<1, 32, 0, stream>>>(static_cast<std::int32_t*>(out.data), alt,
                                                     flag, configs, replacements);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain, std::int32_t columns) {
    return make_sampling_workspace_layout(token_domain, columns).bytes;
}

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t batch             = logits.ne[1];
    const auto* positions                = static_cast<const std::int32_t*>(logical_positions.data);
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, batch);
    if (!layout.multiblock) {
        sample_row_kernel<<<static_cast<unsigned int>(batch), kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            configs, positions, purpose, token_domain, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(batch));
    sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), configs, token_domain, physical_rows,
        scratch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(batch));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
        partial_blocks, groups, scratch);
    CUDA_CHECK(cudaGetLastError());
}

void increment_token_counts_launch(const Tensor& token_ids, Tensor& token_counts,
                                   cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int count      = token_ids.ne[0];
    increment_token_counts_kernel<<<div_up(count, kBlock), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(token_ids.data), count,
        static_cast<std::int32_t*>(token_counts.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
