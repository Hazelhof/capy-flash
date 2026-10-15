#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Qwen Sparse Attention (QSA) of Qwen3.8-Flash-Next (`qwen4_exp`): an indexer selects at most
 * `budget` cached tokens in whole blocks of `ratio` tokens, plus the incomplete tail, and the
 * attention layer runs grouped-query softmax attention over exactly that selection.
 *
 * The three Ops below are the persistent append, the selection and the selected-key attention.
 * Projections, Q/K norms, the attention RoPE and the output gate belong to the caller.
 *
 * Indexer geometry. The indexer has `index_heads` query heads of `index_dim`=128 and one key
 * head. The first `rotary_dim` features of indexer queries and pooled keys rotate with the
 * rotate-half convention: frequency pair i (rotating features i and i+rotary_dim/2) uses
 * angle = FP32(FP32(position) * inv_freq_i) with inv_freq_i = FP32(1 /
 * FP32(theta^(2i/rotary_dim))), the Hugging Face FP32 table and FP32 product, where position is
 * axis `pair_axes[i]` of the token's three multimodal RoPE positions. The FP32 angle is part of the
 * definition: at position 2^16 one FP32 ulp of the angle is ~0.008 rad, far above the other
 * rounding effects. Text tokens carry three equal positions. Qwen3.8-Flash-Next uses interleaved
 * MRoPE sections [11,11,10], i.e. pair_axes[i]=i%3.
 */
inline constexpr std::int32_t kQsaIndexKeyDim = 128;
/** Index record per cached token: 128 raw BF16 key words, three I32 positions, two pad words. */
inline constexpr std::int32_t kQsaIndexRecordWords = 136;
/** Pooled block-key plane: each of a block's `ratio`=4 token slots holds 32 of its 128 words. */
inline constexpr std::int32_t kQsaBlockKeyWords = 32;

struct QsaGeometry {
    std::int32_t query_heads = 0;
    std::int32_t kv_heads    = 0;
    std::int32_t head_dim    = 0;
    std::int32_t index_heads = 0;
    std::int32_t index_dim   = 0;
    std::int32_t budget      = 0; // tokens; budget / ratio whole blocks are kept
    std::int32_t ratio       = 0;
    std::int32_t rotary_dim  = 0;
    float rope_theta         = 0.0F;
    float rms_norm_eps       = 0.0F;
    std::array<std::uint8_t, 64> pair_axes{};

    [[nodiscard]] constexpr std::int32_t block_topk() const noexcept { return budget / ratio; }

    [[nodiscard]] constexpr std::int32_t max_selected() const noexcept {
        return budget + ratio - 1;
    }
};

/** The registered Qwen3.8-Flash-Next geometry: 24/2 heads of 256, indexer 4x128, 2048 in 4s. */
[[nodiscard]] QsaGeometry qwen3_8_flash_next_qsa_geometry();

/**
 * Persistent QSA index planes of one attention layer, paged like the K/V planes (64 tokens per
 * page, page-major). `records` is BF16 [136,64,1,pages]; `block_keys` is BF16 [32,64,1,pages].
 * `block_tables` is I32 [table_stride, rows]; it may be the K/V table itself. Both planes are
 * opaque Op state: only qsa_append writes them and only qsa_select reads them.
 */
struct QsaIndexPlanes {
    Tensor records;
    Tensor block_keys;
    Tensor block_tables;
};

/**
 * Append attention K/V and indexer state for `W` columns of `B` lanes.
 *
 * `k`/`v` are contiguous BF16 [256, kv_heads, W, B] after the attention key norm and RoPE;
 * `index_keys` is contiguous BF16 [128, W, B] raw indexer keys (no norm, no RoPE);
 * `rope_positions` is I32 [W*B, 1|3] axis-major (one column means all three axes are equal);
 * `cache_positions` is I32 [W, B]; `table_rows` I32 [B] selects each lane's block-table row and
 * `valid_columns` (empty or I32 [B]) limits each lane to its first columns. `key_norm` is the
 * BF16 [128] zero-centred indexer key norm weight.
 *
 * K/V storage is byte-identical to kv_cache_append for every cache profile (BF16 K and FP16 V;
 * FP8-row, INT8-G64, NVFP4-G16 and K8V4 with their Hadamard preparation). The raw index key and
 * the token's three positions are stored exactly. After every column of the call is stored, each
 * valid column whose position p completes a block (p % ratio == ratio-1) publishes that block's
 * pooled key
 *
 *   kbar_b = RoPE(offset_rmsnorm(BF16(mean_{j<ratio} raw[ratio*b+j]), key_norm), pos(ratio*b))
 *
 * rounded once to BF16 into `block_keys`, where offset_rmsnorm(x,w) = x/sqrt(mean(x^2)+eps)*(1+w).
 * The pooled mean is rounded to BF16 as upstream does; the RMSNorm and RoPE run in FP32. A block
 * is therefore published exactly when its last token is appended; re-appending any of its tokens
 * (speculative rollback) re-publishes it because tokens are appended in position order.
 * Every addressed row is overwritten and no unrelated row is written. Inputs and every plane are
 * pairwise non-overlapping. The Op owns no allocation, frontier or commit authority.
 */
void qsa_append(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                const Tensor& rope_positions, const Tensor& cache_positions,
                const Tensor& valid_columns, const Tensor& table_rows, const Tensor& key_norm,
                const QsaGeometry& geometry, const PagedKVBatchLayerView& kv,
                const QsaIndexPlanes& index, cudaStream_t stream);

/**
 * Transient capacity of qsa_select for every call of at most `columns` = W*B columns whose
 * visible extents do not exceed `max_visible_keys`.
 */
[[nodiscard]] std::size_t qsa_select_workspace_bytes(const QsaGeometry& geometry,
                                                     std::uint32_t max_visible_keys,
                                                     std::int32_t width, std::int32_t lanes);

/**
 * Select the visible cache indices of every query column.
 *
 * `index_query` is contiguous BF16 [index_heads*128, W, B] raw indexer query rows,
 * `query_rope_positions` is I32 [W*B, 1|3] and `query_norm` the BF16 [128] zero-centred weight.
 * Column (w,b) sees cache indices [0, n) with n = cache_positions[w,b] + 1 and m = floor(n/ratio)
 * complete blocks. With q_h = RoPE(offset_rmsnorm(index_query_h, query_norm), query position) and
 * kbar_b the published pooled key of qsa_append,
 *
 *   score_b = sum_h relu(q_h . kbar_b) / sqrt(128).
 *
 * When m <= block_topk every visible index [0,n) is selected in order. Otherwise the block_topk
 * highest-scoring blocks (the lower block index wins an exact tie) are expanded to their tokens in
 * ascending block order, followed by the incomplete tail [ratio*m, n). `selected` is I32
 * [max_selected, W, B]; `counts` (I32 [W,B]) receives the number of written indices. Columns
 * beyond `valid_columns` produce count zero. Entries past a count are unspecified.
 *
 * The oracle evaluates q_h, kbar_b (from the raw keys, with the BF16 pooled mean) and the scores in
 * FP64; a production selection may differ from it only at near ties of the selection cut, because
 * the published keys are BF16 and the scores accumulate in FP32.
 *
 * `max_visible_keys` is a host launch/workspace promise bounding every live n; it fixes the
 * launch extents, so calls with fixed tensor views stay CUDA Graph update-compatible across
 * envelopes. No device metadata is copied to the host.
 */
void qsa_select(const Tensor& index_query, const Tensor& query_rope_positions,
                const Tensor& cache_positions, const Tensor& valid_columns,
                const Tensor& table_rows, const Tensor& query_norm, const QsaIndexPlanes& index,
                const QsaGeometry& geometry, std::uint32_t max_visible_keys,
                WorkspaceArena& workspace, Tensor& selected, Tensor& counts, cudaStream_t stream);

/**
 * Softmax attention restricted to selected cache indices.
 *
 * `q` is contiguous BF16 [256, query_heads, W, B] after norm and RoPE; query head h reads KV head
 * floor(h / (query_heads/kv_heads)) with at most 16 query heads per KV head. `selected`/`counts`
 * come from qsa_select. For each column and query head the oracle is the scaled dot-product
 * softmax over exactly the selected keys, reading K/V at their storage boundary as the dense
 * causal_softmax_attention contract defines it (FP16 V for the BFloat16 profile; for a quantized
 * profile each stored code times its scale in the stored basis, with q rotated like K and the
 * output of a rotated V rotated back). `out` has the shape of q; a column with count zero writes
 * exact zeros. `workspace` holds qsa_attention_workspace_bytes for the call's W*B columns.
 */
void qsa_attention(const Tensor& q, const Tensor& selected, const Tensor& counts,
                   const Tensor& table_rows, const PagedKVBatchLayerView& kv, float scale,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/** Workspace of qsa_attention for every call of at most `columns` columns. */
[[nodiscard]] std::size_t qsa_attention_workspace_bytes(const QsaGeometry& geometry,
                                                        std::int32_t columns);

} // namespace ninfer::ops
