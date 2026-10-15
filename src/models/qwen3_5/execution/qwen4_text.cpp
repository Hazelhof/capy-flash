// Qwen4Exp (Qwen3.8-Flash-Next) entry points of TextContext: chunked Prefill (with the MTP prompt
// KV and first proposals), ordinary decode, speculative target verification and the MTP layer's
// decode, bridge and proposal calls at one device; chunked Prefill, ordinary decode and the MTP
// calls (verification, MTP layer, proposal and prefix-reuse bridge) at tensor-parallel width 2.
// The forward itself is execution/qwen4_forward.h; the round schedule is
// program/speculative/mtp.cpp.
//
// Every Qwen4Exp call carries the wide residual (S*H) as its hidden: the output head reads it
// through the Text output mixer, the MTP stem consumes it whole.
#include "models/qwen3_5/execution/text.h"

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "models/qwen3_5/execution/qwen4_draft_head.h"
#include "models/qwen3_5/execution/tp.h"
#include "models/qwen3_5/execution/tp2/workspace_split.h"
#include "models/qwen3_5/program/speculative/mtp_alignment.h"

#include "core/device_scope.h"
#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                   const char* label) {
    bool ok = t.dtype == dtype && t.is_contiguous() && t.data != nullptr;
    int i   = 0;
    for (const std::int32_t dim : shape) { ok = ok && t.ne[i++] == dim; }
    for (; i < 4; ++i) { ok = ok && t.ne[i] == 1; }
    if (!ok) { throw std::invalid_argument(std::string("Qwen4Exp ") + label + " shape mismatch"); }
}

// T42: one prompt chunk wider than the Program's arena runs on workspace memory the expert
// backend lends (Qwen4RoutedExperts::prompt_workspace). The TextContext's arena object belongs to
// the Program; for the chunk it is moved aside and replaced by a non-owning arena over the lent
// span, and restored afterwards (also on unwinding, after the window helper waited for the device).
class ArenaLoan {
public:
    ArenaLoan() = default;
    ~ArenaLoan() { restore(); }
    ArenaLoan(const ArenaLoan&)            = delete;
    ArenaLoan& operator=(const ArenaLoan&) = delete;

    void lend(WorkspaceArena& arena, DeviceSpan span) {
        if (target_ != nullptr) { throw std::logic_error("Qwen4Exp arena loan already taken"); }
        saved_.emplace(std::move(arena));
        target_ = &arena;
        arena   = WorkspaceArena(span);
    }
    void restore() noexcept {
        if (target_ == nullptr) { return; }
        *target_ = std::move(*saved_);
        saved_.reset();
        target_ = nullptr;
    }

private:
    WorkspaceArena* target_ = nullptr;
    std::optional<WorkspaceArena> saved_;
};

// The carried hidden of a prompt chunk: the persistent prefill_hidden when it holds the chunk,
// else (a chunk on lent workspace, T42) a matrix of the chunk's arena.
Tensor chunk_hidden(const Tensor& prefill_hidden, WorkspaceArena& work, std::int32_t rows,
                    std::int32_t len) {
    if (prefill_hidden.data != nullptr && len <= prefill_hidden.ne[1]) {
        return prefill_hidden.slice(1, 0, len);
    }
    return work.alloc(DType::BF16, {rows, len});
}

// A chunk whose hidden is not prefill_hidden leaves its last column in prefill_hidden's last
// column: the Program's copy_tail reads the final chunk's last column there (T42).
void keep_last_column(const Tensor& xf, const Tensor& prefill_hidden, cudaStream_t stream) {
    if (prefill_hidden.data == nullptr || xf.data == prefill_hidden.data) { return; }
    const Tensor last   = xf.slice(1, xf.ne[1] - 1, 1);
    const Tensor target = prefill_hidden.slice(1, prefill_hidden.ne[1] - 1, 1);
    CUDA_CHECK(::ninfer::memcpy_device_async(target.data, last.data, last.bytes(),
                               stream));
}

} // namespace

const Qwen4Runtime& TextContext::qwen4_runtime() const {
    if (qwen4_runtime_ == nullptr || qwen4_runtime_->experts == nullptr) {
        throw std::logic_error("Qwen4Exp execution has no bound Program runtime");
    }
    return *qwen4_runtime_;
}

PrefillChunkResult TextContext::qwen4_prefill(std::span<const int> ids,
                                              const TextPrefill* text_prefill,
                                              bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (tp2()) { return qwen4_prefill_tp2(ids, text_prefill, finalize_at_end); }
    const Qwen4Runtime& runtime = qwen4_runtime();
    cudaStream_t s              = ctx_.stream;
    const auto T                = static_cast<std::int64_t>(ids.size());
    const std::uint32_t base    = text_kv_base_;
    // The n-gram embedding hashes the tokens before each column, so the prompt must be whole.
    if (text_prefill == nullptr || base != text_prefill->begin ||
        text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
        throw std::invalid_argument("Qwen4Exp prefill requires its full prompt");
    }
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        (qwen4_mtp_ == nullptr ||
         mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0]))) {
        throw std::logic_error("Qwen4Exp MTP prefill has no MTP layer or exceeds the draft window");
    }
    if (text_kv_base_ == 0) { rope_delta_ = 0; }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);
    const int base_i              = static_cast<int>(base);
    const std::int64_t split_abs  = prefill_split_frontier_;
    const bool has_split          = split_abs > base_i && split_abs <= base_i + T;
    const int split_rel           = has_split ? static_cast<int>(split_abs - base_i) : -1;

    int len = static_cast<int>(std::min<std::int64_t>(prefill_chunk_, T));
    if (split_rel > 0 && len > split_rel) { len = split_rel; }
    const bool is_last = finalize_at_end && len == T;
    nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                  static_cast<std::uint64_t>(len));
    const std::int32_t wide_rows = dimension(config_.residual_width());
    const std::int32_t H         = dimension(config_.hidden_size);
    work_.reset();
    // One expert window per chunk (with its MTP prompt calls): the backend observes the chunk's
    // routing once it completed. A chunk wider than the Program's arena runs on lent memory.
    ArenaLoan loan;
    const bool lent_arena = len > runtime.experts->prompt_arena_columns();
    runtime.experts->prepare_prompt(len);
    run_expert_window(*runtime.experts, s, Qwen4Window::Prompt, [&] {
        if (lent_arena) {
            const cudaStream_t streams[1] = {s};
            loan.lend(work_, runtime.experts->prompt_workspace(streams)[0]);
        }
        Tensor ids_device = work_.alloc(DType::I32, {len});
        CUDA_CHECK(cudaMemcpyAsync(ids_device.data, ids.data(), ids_device.bytes(),
                                   cudaMemcpyHostToDevice, s));
        Tensor positions = work_.alloc(DType::I32, {len});
        ops::fill_i32_positions(positions, base_i, s);
        Tensor rope_positions = positions;
        if (rope_delta_ != 0) {
            rope_positions = work_.alloc(DType::I32, {len});
            ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
        }
        Tensor ple_embedding;
        if (config_.ple) {
            if (runtime.ple_gather == nullptr || runtime.ple_host == nullptr) {
                throw std::logic_error("Qwen4Exp prefill has no PLE gather staging");
            }
            const std::uint32_t width = runtime.ple_gather->width();
            const std::span<const std::int32_t> history(text_prefill->token_ids.data(),
                                                        text_prefill->token_ids.size());
            // The staging buffer is reused per chunk; the previous chunk ended in a device
            // synchronization, so its upload has completed.
            runtime.ple_gather->gather(
                history, base, static_cast<std::size_t>(len),
                std::span<std::uint16_t>(runtime.ple_host, static_cast<std::size_t>(width) * len));
            ple_embedding = work_.alloc(DType::BF16, {static_cast<std::int32_t>(width), len});
            CUDA_CHECK(cudaMemcpyAsync(ple_embedding.data, runtime.ple_host, ple_embedding.bytes(),
                                       cudaMemcpyHostToDevice, s));
        }
        // The carried hidden of every prompt column: the committed wide residual.
        Tensor xf = chunk_hidden(prefill_hidden_, work_, wide_rows, len);
        const Qwen4Columns columns{.width            = len,
                                   .batch            = 1,
                                   .cache_positions  = &positions,
                                   .rope_positions   = &rope_positions,
                                   .valid_columns    = nullptr,
                                   .table_rows       = &io_.text_kv_table_row,
                                   .max_visible      = static_cast<std::uint32_t>(base_i + len),
                                   .prefill          = true,
                                   .source_slot      = linear_state_source_slot_,
                                   .destination_slot = linear_state_destination_slot_,
                                   .first_position   = base_i};
        const Qwen4State state{.kv = batch_text_kv_, .linear = &state_, .runtime = &runtime};
        if (state.kv == nullptr || !state.kv->has_qsa_index()) {
            throw std::logic_error("Qwen4Exp prefill requires the Main KV cache with QSA planes");
        }
        Qwen4Forward<WorkspaceArena> forward(config_, parameters_, work_, s, ctx_.execution_view(),
                                             columns, state, len, len);
        forward.run(ids_device, config_.ple ? &ple_embedding : nullptr, xf);

        if (is_last) {
            auto head_scope = work_.scope();
            Tensor x        = work_.alloc(DType::BF16, {H, 1});
            forward.head(parameters_.qwen4->head, xf.slice(1, len - 1, 1), x);
            Tensor logits = io_.logits.slice(1, 0, 1);
            ops::linear(x, lm_head_->weight, logits, lm_head_->policy, work_, s);
            ops::set_i32_scalar(io_.pos, base_i + static_cast<int>(T), s);
            ops::set_i32_scalar(io_.rope_pos, base_i + static_cast<int>(T) + rope_delta_, s);
            if (sampling_config_ != nullptr) {
                ops::sample(logits, io_.token,
                            dimension(parameters_.model.resources().public_token_count),
                            sampling_config_, io_.pos, ops::kSamplePurposePrefill, work_, s);
            } else {
                ops::argmax(logits, io_.token,
                            dimension(parameters_.model.resources().public_token_count), s);
            }
        }

        if (prepare_mtp_prompt && runtime.taps != nullptr && runtime.taps->wants_mtp_logits() &&
            base == 0) {
            runtime.taps->capture_tokens(std::span<const std::int32_t>(
                text_prefill->token_ids.data(), text_prefill->token_ids.size()));
        }
        if (prepare_mtp_prompt) {
            // MTP alignment of this chunk: hidden h_p and position p with token x_(p+1); the last
            // prompt column takes the generated token (qwen3_5-model.md#prefill-decode-and-mtp).
            const auto alignment_tokens =
                static_cast<std::uint32_t>(text_prefill->token_ids.size());
            const qwen3_5::MtpAlignmentWindow window =
                qwen3_5::plan_mtp_alignment_window(alignment_tokens, base,
                                                   static_cast<std::uint32_t>(len));
            const int prompt_columns =
                len - static_cast<int>(window.final_column_uses_generated_token);
            auto mtp_scope           = work_.scope();
            Tensor mtp_ids           = work_.alloc(DType::I32, {len});
            if (prompt_columns != 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    mtp_ids.data, text_prefill->token_ids.data() + window.shifted_embedding_begin,
                    static_cast<std::size_t>(prompt_columns) * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice, s));
            }
            if (window.final_column_uses_generated_token) {
                CUDA_CHECK(::ninfer::memcpy_device_async(mtp_ids.slice(0, len - 1, 1).data, io_.token.data,
                                           sizeof(std::int32_t), s));
            }
            qwen4_mtp_prefill(mtp_ids, xf, positions, rope_positions,
                              static_cast<std::uint32_t>(base_i + len),
                              is_last && mtp_proposal_extent_ != 0);
        }

        if (split_rel > 0 && len == split_rel && rewrite_checkpoint_hidden_output_ != nullptr) {
            const Tensor checkpoint = xf.slice(1, len - 1, 1);
            CUDA_CHECK(::ninfer::memcpy_device_async(rewrite_checkpoint_hidden_output_->data, checkpoint.data,
                                       checkpoint.bytes(), s));
        }
        keep_last_column(xf, prefill_hidden_, s);
        prefill_split_frontier_ = -1;
        timing.begin_wait();
    }, [&] {
        ctx_.synchronize();
        timing.end_wait();
    });
    loan.restore();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(len),
                              .finalized        = finalize_at_end && len == T,
                              .timing           = timing.finish()};
}

// MTP KV of one prompt chunk: every column is appended to the MTP cache; with `propose` the last
// column (the generated token) runs the whole MTP layer and drafts mtp_proposal_extent_ tokens
// autoregressively into io_.mtp (draft_tokens, ar_hidden, position).
void TextContext::qwen4_mtp_prefill(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, const Tensor& rope_positions,
                                    std::uint32_t max_visible, bool propose) {
    if (batch_mtp_kv_ == nullptr || qwen4_mtp_ == nullptr || !batch_mtp_kv_->has_qsa_index()) {
        throw std::logic_error("Qwen4Exp MTP prefill requires the MTP layer and its QSA cache");
    }
    const Qwen4Runtime& runtime = qwen4_runtime();
    cudaStream_t s              = ctx_.stream;
    const std::int32_t T        = ids.ne[0];
    const std::int32_t append   = propose ? T - 1 : T;
    const Qwen4State state{.kv = batch_mtp_kv_, .linear = &state_, .runtime = &runtime};
    nvtx::ScopedRange range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(T));
    const auto columns_of = [&](std::int32_t first, std::int32_t count, const Tensor& pos,
                                const Tensor& rope) {
        return Qwen4Columns{.width           = count,
                            .batch           = 1,
                            .cache_positions = &pos,
                            .rope_positions  = &rope,
                            .valid_columns   = nullptr,
                            .table_rows      = &io_.backend_kv_table_row,
                            .max_visible     = max_visible,
                            .prefill         = true,
                            .first_position  = first};
    };
    if (runtime.taps != nullptr && runtime.taps->wants_mtp_logits()) {
        // Teacher forcing (validation, eager): every column runs the whole layer, one at a time
        // (causal, so equal to the batched append), and its draft logits are captured.
        for (std::int32_t c = 0; c < append; ++c) {
            auto scope        = work_.scope();
            const Tensor pos  = positions.slice(0, c, 1);
            const Tensor rope = rope_positions.slice(0, c, 1);
            Tensor out  = work_.alloc(DType::BF16, {dimension(config_.residual_width()), 1});
            Tensor draft = work_.alloc(DType::I32, {1});
            {
                Qwen4Forward<WorkspaceArena> forward(config_, parameters_, work_, s,
                                                     ctx_.execution_view(),
                                                     columns_of(c, 1, pos, rope), state, 1, 1);
                forward.run_mtp(*qwen4_mtp_, ids.slice(0, c, 1), nullptr, hidden.slice(1, c, 1),
                                out, false);
            }
            Tensor logits = io_.logits.slice(1, 0, 1);
            qwen4_mtp_propose(out, logits, draft);
            std::int32_t position = 0;
            CUDA_CHECK(cudaMemcpyAsync(&position, pos.data, sizeof(position),
                                       cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            runtime.taps->capture_mtp_logits(logits, position, s);
        }
    } else if (append > 0) {
        auto scope        = work_.scope();
        const Tensor pos  = positions.slice(0, 0, append);
        const Tensor rope = rope_positions.slice(0, 0, append);
        Tensor scratch    = work_.alloc(DType::BF16, {dimension(config_.residual_width()), append});
        Qwen4Forward<WorkspaceArena> forward(config_, parameters_, work_, s, ctx_.execution_view(),
                                             columns_of(0, append, pos, rope), state, append,
                                             append);
        forward.run_mtp(*qwen4_mtp_, ids.slice(0, 0, append), nullptr,
                        hidden.slice(1, 0, append), scratch, true);
    }
    if (!propose) { return; }
    const Tensor pos  = positions.slice(0, T - 1, 1);
    const Tensor rope = rope_positions.slice(0, T - 1, 1);
    {
        Qwen4Forward<WorkspaceArena> forward(config_, parameters_, work_, s, ctx_.execution_view(),
                                             columns_of(T - 1, 1, pos, rope), state, 1, 1);
        forward.run_mtp(*qwen4_mtp_, ids.slice(0, T - 1, 1), nullptr, hidden.slice(1, T - 1, 1),
                        io_.mtp->ar_hidden, false);
    }
    Tensor logits = io_.logits.slice(1, 0, 1);
    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
    qwen4_mtp_propose(io_.mtp->ar_hidden, logits, draft0);
    // Autoregressive drafts: draft i at position (frontier + i) from draft i-1 and the previous
    // MTP hidden.
    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, static_cast<std::int32_t>(max_visible), s);
    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
        auto step_scope    = work_.scope();
        Tensor previous    = io_.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next        = io_.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden = work_.alloc(DType::BF16, {dimension(config_.residual_width()), 1});
        const auto visible = max_visible + static_cast<std::uint32_t>(i);
        qwen4_mtp_forward_batch(previous, io_.mtp->ar_hidden, ar_position, {visible, visible},
                                next_hidden, 0, &logits, &next, nullptr, nullptr);
        CUDA_CHECK(::ninfer::memcpy_device_async(io_.mtp->ar_hidden.data, next_hidden.data,
                                   io_.mtp->ar_hidden.bytes(), s));
        ops::increment_i32_scalar(ar_position, s);
    }
}

void TextContext::qwen4_ordinary_decode(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& source_slots,
                                        const Tensor& destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    if (mtp_enabled()) { throw std::logic_error("Qwen4Exp MTP decodes through the MTP round"); }
    if (tp2()) {
        qwen4_ordinary_decode_tp2(ids, cache_positions, rope_positions, kv_table_rows, source_slots,
                                  destination_slots, envelope, hidden, logits);
        return;
    }
    const Qwen4Runtime& runtime = qwen4_runtime();
    const std::int32_t batch = ids.ne[0];
    cudaStream_t stream      = ctx_.stream;
    Tensor rows;
    if (config_.ple) {
        if (runtime.decode_ple_rows.data == nullptr || runtime.decode_ple_rows.ne[1] < batch) {
            throw std::logic_error("Qwen4Exp decode has no PLE rows");
        }
        rows = runtime.decode_ple_rows.slice(1, 0, batch);
    }
    const Qwen4Columns columns{.width             = 1,
                               .batch             = batch,
                               .cache_positions   = &cache_positions,
                               .rope_positions    = &rope_positions,
                               .valid_columns     = nullptr,
                               .table_rows        = &kv_table_rows,
                               .max_visible       = envelope.max_visible_keys,
                               .prefill           = false,
                               .source_slots      = &source_slots,
                               .destination_slots = &destination_slots};
    const Qwen4State state{.kv = batch_text_kv_, .linear = &state_, .runtime = &runtime};
    if (state.kv == nullptr || !state.kv->has_qsa_index()) {
        throw std::logic_error("Qwen4Exp decode requires the Main KV cache with QSA planes");
    }
    work_.reset();
    {
        Qwen4Forward<WorkspaceArena> forward(config_, parameters_, work_, stream,
                                             ctx_.execution_view(), columns, state, batch, batch);
        forward.run(ids, config_.ple ? &rows : nullptr, hidden);
        Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), batch});
        forward.head(parameters_.qwen4->head, hidden, x);
        ops::linear(x, lm_head_->weight, logits, lm_head_->policy, work_, stream);
    }
    work_.reset();
}

// Target verification of W = K+1 columns of B lanes (column 0 the anchor, then the drafts):
// GDN records ReplaySSM transitions and PLE its convolution inputs (folded at commit), QSA
// appends in place; hidden receives the wide residuals, logits/target_tokens the Text head.
void TextContext::qwen4_target_verify(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens) {
    const Qwen4Runtime& runtime = qwen4_runtime();
    const std::int32_t W        = ids.ne[0];
    const std::int32_t B        = ids.ne[1];
    const std::int32_t T        = W * B;
    if (W < 2 || B <= 0 || B > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("Qwen4Exp verification window is outside [2,K+1]x[1,8]");
    }
    if (gdn_state_action_ != GdnStateAction::RecordForReplay || replay_records_ == nullptr) {
        throw std::logic_error("Qwen4Exp verification requires ReplaySSM records");
    }
    const std::int32_t wide_rows = dimension(config_.residual_width());
    require_shape(ids, DType::I32, {W, B}, "verify ids");
    require_shape(cache_positions, DType::I32, {W, B}, "verify positions");
    require_shape(rope_positions, DType::I32, {W, B}, "verify RoPE positions");
    require_shape(valid_columns, DType::I32, {B}, "verify valid columns");
    require_shape(kv_table_rows, DType::I32, {B}, "verify KV rows");
    require_shape(source_slots, DType::I32, {B}, "verify state slots");
    require_shape(hidden, DType::BF16, {wide_rows, W, B}, "verify hidden");
    require_shape(logits, DType::BF16, {dimension(config_.vocab_size), W, B}, "verify logits");
    require_shape(target_tokens, DType::I32, {W, B}, "verify tokens");
    Tensor rows;
    if (config_.ple) {
        if (runtime.decode_ple_rows.data == nullptr || runtime.decode_ple_rows.ne[1] < T) {
            throw std::logic_error("Qwen4Exp verification has no PLE rows");
        }
        rows = runtime.decode_ple_rows.slice(1, 0, T);
    }
    const Qwen4Columns columns{.width             = W,
                               .batch             = B,
                               .cache_positions   = &cache_positions,
                               .rope_positions    = &rope_positions,
                               .valid_columns     = &valid_columns,
                               .table_rows        = &kv_table_rows,
                               .max_visible       = envelope.max_visible_keys,
                               .prefill           = false,
                               .source_slots      = &source_slots,
                               .destination_slots = &source_slots,
                               .replay_records    = replay_records_};
    const Qwen4State state{.kv = batch_text_kv_, .linear = &state_, .runtime = &runtime};
    if (state.kv == nullptr || !state.kv->has_qsa_index()) {
        throw std::logic_error("Qwen4Exp verification requires the Main KV cache with QSA planes");
    }
    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        Tensor flat_ids    = ids.view({T});
        Tensor flat_hidden = hidden.view({wide_rows, T});
        Qwen4Forward<WorkspaceArena> forward(config_, parameters_, work_, stream,
                                             ctx_.execution_view(), columns, state, T, T);
        forward.run(flat_ids, config_.ple ? &rows : nullptr, flat_hidden);
        Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), T});
        forward.head(parameters_.qwen4->head, flat_hidden, x);
        Tensor flat_logits = logits.view({dimension(config_.vocab_size), T});
        ops::linear(x, lm_head_->weight, flat_logits, lm_head_->policy, work_, stream);
        Tensor flat_tokens = target_tokens.view({T});
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), stream);
    }
    work_.reset();
}

// The MTP layer over the round's W alignment columns of B lanes (the verified tokens shifted by
// one with the target's wide residuals), or one autoregressive column per lane.
void TextContext::qwen4_mtp_forward_decode(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    if (qwen4_mtp_ == nullptr || batch_mtp_kv_ == nullptr || !batch_mtp_kv_->has_qsa_index()) {
        throw std::logic_error("Qwen4Exp MTP decode requires the MTP layer and its QSA cache");
    }
    const Qwen4Runtime& runtime  = qwen4_runtime();
    const std::int32_t W         = ids.ne[0];
    const std::int32_t B         = ids.ne[1];
    const std::int32_t T         = W * B;
    const std::int32_t wide_rows = dimension(config_.residual_width());
    if (W <= 0 || W > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || B <= 0 ||
        B > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("Qwen4Exp MTP decode shape is outside the supported domain");
    }
    require_shape(ids, DType::I32, {W, B}, "MTP decode ids");
    require_shape(hidden, DType::BF16, {wide_rows, W, B}, "MTP decode target hidden");
    require_shape(cache_positions, DType::I32, {W, B}, "MTP decode positions");
    require_shape(rope_positions, DType::I32, {W, B}, "MTP decode RoPE positions");
    require_shape(valid_columns, DType::I32, {B}, "MTP decode valid columns");
    require_shape(kv_table_rows, DType::I32, {B}, "MTP decode KV rows");
    require_shape(mtp_hidden, DType::BF16, {wide_rows, W, B}, "MTP decode hidden");
    // Only the accepted column's output is read (speculative_select_accepted_hidden: column
    // valid_columns[b] - 1 of the alignment, the single column of an AR step), so the MoE routes
    // only that column; the others still append their MTP K/V.
    const Qwen4Columns columns{.width               = W,
                               .batch               = B,
                               .cache_positions     = &cache_positions,
                               .rope_positions      = &rope_positions,
                               .valid_columns       = &valid_columns,
                               .moe_last_valid_only = true,
                               .table_rows          = &kv_table_rows,
                               .max_visible         = envelope.max_visible_keys,
                               .prefill             = false};
    const Qwen4State state{.kv = batch_mtp_kv_, .linear = &state_, .runtime = &runtime};
    nvtx::ScopedRange range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(T));
    auto scope = work_.scope();
    Qwen4Forward<WorkspaceArena> forward(config_, parameters_, work_, ctx_.stream,
                                         ctx_.execution_view(), columns, state, T, T);
    Tensor out = mtp_hidden.view({wide_rows, T});
    forward.run_mtp(*qwen4_mtp_, ids.view({T}), nullptr, hidden.view({wide_rows, T}), out, false);
}

// One sequence's MTP forward over T consecutive positions (the prefix-reuse bridge and the
// autoregressive drafts of the prompt's last column), on the prefill KV row of the MTP cache.
void TextContext::qwen4_mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                          const Tensor& positions,
                                          ops::CausalAttentionExecutionEnvelope envelope,
                                          Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                          Tensor* draft_token,
                                          const Tensor* explicit_rope_positions,
                                          const Tensor* input_embeddings) {
    if (qwen4_mtp_ == nullptr || batch_mtp_kv_ == nullptr || !batch_mtp_kv_->has_qsa_index()) {
        throw std::logic_error("Qwen4Exp MTP forward requires the MTP layer and its QSA cache");
    }
    if (input_embeddings != nullptr) {
        throw std::invalid_argument("Qwen4Exp MTP takes token embeddings only (no image input)");
    }
    const Qwen4Runtime& runtime  = qwen4_runtime();
    const std::int32_t T         = ids.ne[0];
    const std::int32_t wide_rows = dimension(config_.residual_width());
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_ || logits_column >= T) {
        throw std::invalid_argument("Qwen4Exp MTP batch columns are outside [1,prefill_chunk]");
    }
    require_shape(ids, DType::I32, {T}, "MTP ids");
    require_shape(positions, DType::I32, {T}, "MTP positions");
    require_shape(hidden, DType::BF16, {wide_rows, T}, "MTP hidden");
    require_shape(mtp_hidden, DType::BF16, {wide_rows, T}, "MTP output hidden");
    cudaStream_t s = ctx_.stream;
    auto scope     = work_.scope();
    Tensor rope;
    if (explicit_rope_positions != nullptr) {
        // The bridge passes one three-axis position; the Text RoPE of this architecture is 1-D.
        rope = explicit_rope_positions->ne[1] == 3
                   ? explicit_rope_positions->slice(1, 0, 1).view({T})
                   : explicit_rope_positions->view({T});
    } else {
        rope = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, rope, s);
    }
    const Qwen4Columns columns{.width           = T,
                               .batch           = 1,
                               .cache_positions = &positions,
                               .rope_positions  = &rope,
                               .valid_columns   = nullptr,
                               .table_rows      = &io_.backend_kv_table_row,
                               .max_visible     = envelope.max_visible_keys,
                               .prefill         = true};
    const Qwen4State state{.kv = batch_mtp_kv_, .linear = &state_, .runtime = &runtime};
    {
        Qwen4Forward<WorkspaceArena> forward(config_, parameters_, work_, s, ctx_.execution_view(),
                                             columns, state, T, T);
        forward.run_mtp(*qwen4_mtp_, ids, nullptr, hidden, mtp_hidden, false);
    }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("Qwen4Exp MTP logits and draft outputs are required");
        }
        Tensor column = mtp_hidden.slice(1, logits_column, 1);
        qwen4_mtp_propose(column, *logits, *draft_token);
    }
}

void TextContext::qwen4_logits_from_hidden(const Tensor& hidden, const Tensor* peer_hidden,
                                           Tensor& logits) {
    if (!qwen4()) { throw std::logic_error("qwen4_logits_from_hidden is Qwen4Exp only"); }
    const std::int32_t wide_rows = dimension(config_.residual_width());
    const std::int32_t H         = dimension(config_.hidden_size);
    require_shape(hidden, DType::BF16, {wide_rows, 1}, "retained hidden");
    require_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "retained logits");
    const ScopedCurrentDevice device(ctx_.device);
    if (!tp2()) {
        auto scope = work_.scope();
        Tensor x   = work_.alloc(DType::BF16, {H, 1});
        const Qwen4Columns columns{.width = 1, .batch = 1, .prefill = false};
        const Qwen4State state{.kv = nullptr, .linear = &state_, .runtime = qwen4_runtime_};
        Qwen4Forward<WorkspaceArena> forward(config_, parameters_, work_, ctx_.stream,
                                             ctx_.execution_view(), columns, state, 1, 1);
        forward.head(parameters_.qwen4->head, hidden, x);
        ops::linear(x, lm_head_->weight, logits, lm_head_->policy, work_, ctx_.stream);
        return;
    }
    const ExecutionContext& execution = *tp_->execution;
    const auto ws = workspaces();
    auto scope0   = work_.scope();
    auto scope1   = tp_->work->scope();
    Tensor peer;
    if (peer_hidden != nullptr) {
        require_shape(*peer_hidden, DType::BF16, {wide_rows, 1}, "rank 1 retained hidden");
        peer = *peer_hidden;
    } else {
        peer = tp_->work->alloc(DType::BF16, {wide_rows, 1});
        CUDA_CHECK(cudaEventRecord(tp_->events->inputs_ready(0), ctx_.stream));
        const ScopedCurrentDevice rank1(execution.dev[1]->device);
        CUDA_CHECK(cudaStreamWaitEvent(rank_stream(1), tp_->events->inputs_ready(0), 0));
        CUDA_CHECK(::ninfer::memcpy_device_async(peer.data, hidden.data, peer.bytes(),
                                                 rank_stream(1)));
    }
    std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
    for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        ranks[r]     = Qwen4Rank<WorkspaceArena>{
                .config     = &*shard_config_,
                .parameters = &rank_parameters(rank),
                .work       = ws[r],
                .stream     = rank_stream(rank),
                .execution  = rank_execution_view(rank),
                .device     = rank == 0 ? ctx_.device : execution.dev[1]->device,
                .columns    = {.width = 1, .batch = 1, .prefill = false},
                .state      = {.kv = nullptr, .linear = &rank_state(rank), .runtime = nullptr}};
    }
    Qwen4Forward<WorkspaceArena> forward(config_, ranks, kTensorParallelWidth,
                                         {tp_->execution, tp_->events}, 1, 1);
    const RankTensors wide{hidden, peer};
    RankTensors x{ws[0]->alloc(DType::BF16, {H, 1}), ws[1]->alloc(DType::BF16, {H, 1})};
    forward.head(wide, x);
    logits_tp2(x, logits);
}

// Draft proposal from MTP hiddens: the MTP output mixer, then the Text output head over the full
// vocabulary and argmax, or (T43) the draft shortlist head of Qwen4Runtime::draft_head. (The
// artifact proposal-head hook of TextContext stays refused for this architecture.)
void TextContext::qwen4_mtp_propose(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens,
                                    Tensor* draft_logprobs) {
    if (qwen4_mtp_ == nullptr) { throw std::logic_error("Qwen4Exp proposal has no MTP layer"); }
    const std::int32_t B         = hidden.ne[1];
    const std::int32_t wide_rows = dimension(config_.residual_width());
    require_shape(hidden, DType::BF16, {wide_rows, B}, "MTP proposal hidden");
    require_shape(draft_tokens, DType::I32, {B}, "MTP proposal tokens");
    if (logits.dtype != DType::BF16 || logits.ne[0] != dimension(config_.vocab_size) ||
        logits.ne[1] < B || !logits.is_contiguous()) {
        throw std::invalid_argument("Qwen4Exp MTP proposal logits shape mismatch");
    }
    if (proposal_head_ != nullptr) {
        throw std::logic_error("Qwen4Exp has no trimmed proposal head yet");
    }
    nvtx::ScopedRange range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(B));
    cudaStream_t s      = ctx_.stream;
    auto scope          = work_.scope();
    Tensor x            = work_.alloc(DType::BF16, {dimension(config_.hidden_size), B});
    const Qwen4Columns columns{.width = B, .batch = 1, .prefill = false};
    const Qwen4State state{.kv = nullptr, .linear = &state_, .runtime = qwen4_runtime_};
    Qwen4Forward<WorkspaceArena> forward(config_, parameters_, work_, s, ctx_.execution_view(),
                                         columns, state, B, B);
    forward.head(qwen4_mtp_->head, hidden, x);
    Tensor window = logits.slice(1, 0, B);
    if (qwen4_runtime_ != nullptr && qwen4_runtime_->draft_head != nullptr) {
        draft_head_propose(*qwen4_runtime_->draft_head, x, window, draft_tokens, draft_logprobs,
                           work_, s);
        return;
    }
    ops::linear(x, lm_head_->weight, window, lm_head_->policy, work_, s);
    const std::int32_t vocabulary = dimension(parameters_.model.resources().public_token_count);
    ops::argmax(window, draft_tokens, vocabulary, s);
    if (draft_logprobs != nullptr) {
        ops::target_logprobs(window, draft_tokens, vocabulary, *draft_logprobs, s);
    }
}

// =================================================================================================
// Tensor parallelism (tp == 2)
// =================================================================================================
//
// Both ranks run the same Qwen4Forward schedule in lockstep (qwen4_forward.h): the replicated
// hyper-connection residual, router, indexer and PLE on each rank's own copy of the inputs, the
// rank's QSA/GDN heads and expert partition, and two all-reduces per block. Rank 0 owns sampling
// and the logits (vocabulary-split head gathered to rank 0, logits_tp2); rank 1 reads its own
// copies of the control tensors (prefill: uploaded here; decode: the OrdinaryPeerFrame).

void TextContext::validate_tensor_parallel_qwen4() {
    const TpExecution& tp = *tp_;
    shard_config_         = shard_text_config(config_, kTensorParallelWidth);
    if (mtp_enabled()) {
        // T33: MTP over both ranks (each rank's MTP QSA heads, expert partition, MTP KV).
        if (qwen4_mtp_ == nullptr || !tp.parameters->qwen4_mtp || !tp.mtp_complete() ||
            batch_mtp_kv_ == nullptr || !batch_mtp_kv_->has_qsa_index() ||
            !tp.mtp_cache->has_qsa_index()) {
            throw std::invalid_argument("tensor-parallel Qwen4Exp MTP binding is incomplete");
        }
    }
    if (!tp.parameters->qwen4 || !parameters_.qwen4 ||
        tp.parameters->qwen4->layers.size() != parameters_.qwen4->layers.size() ||
        parameters_.qwen4->layers.size() != config_.layer_types.size()) {
        throw std::invalid_argument(
            "tensor-parallel Qwen4Exp requires both ranks' Qwen4Exp Parameters");
    }
    if (tp.qwen4 == nullptr || tp.qwen4->experts == nullptr) {
        throw std::invalid_argument("tensor-parallel Qwen4Exp has no rank 1 runtime");
    }
    if (config_.full_attention_layers != 0) {
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const qwen3_5::PagedKVCache& cache = rank_text_cache(rank);
            const PagedKVBatchLayerView view   = cache.batch_layer_view(0);
            if (!cache.has_qsa_index() ||
                view.num_kv_heads != dimension(shard_config_->attention->num_key_value_heads) ||
                view.head_dim != dimension(shard_config_->attention->head_dim)) {
                throw std::invalid_argument("tensor-parallel Qwen4Exp KV cache does not hold one "
                                            "rank's KV head with its QSA index planes");
            }
        }
    }
}

const Qwen4Runtime& TextContext::qwen4_peer_runtime() const {
    if (tp_ == nullptr || tp_->qwen4 == nullptr || tp_->qwen4->experts == nullptr) {
        throw std::logic_error("Qwen4Exp tp 2 execution has no bound rank 1 runtime");
    }
    return *tp_->qwen4;
}

PrefillChunkResult TextContext::qwen4_prefill_tp2(std::span<const int> ids,
                                                  const TextPrefill* text_prefill,
                                                  bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    const std::array<const Qwen4Runtime*, 2> runtime{&qwen4_runtime(), &qwen4_peer_runtime()};
    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    cudaStream_t s           = ctx_.stream;
    const auto T             = static_cast<std::int64_t>(ids.size());
    const std::uint32_t base = text_kv_base_;
    if (text_prefill == nullptr || base != text_prefill->begin ||
        text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
        throw std::invalid_argument("Qwen4Exp prefill requires its full prompt");
    }
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    if (text_kv_base_ == 0) { rope_delta_ = 0; }
    // Each rank offsets its own positions by its own copy of the delta.
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);
    {
        const ScopedCurrentDevice peer(execution.dev[1]->device);
        Tensor peer_rope_delta = tp_->rope_delta;
        ops::set_i32_scalar(peer_rope_delta, rope_delta_, rank_stream(1));
    }
    const int base_i             = static_cast<int>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split         = split_abs > base_i && split_abs <= base_i + T;
    const int split_rel          = has_split ? static_cast<int>(split_abs - base_i) : -1;

    int len = static_cast<int>(std::min<std::int64_t>(prefill_chunk_, T));
    if (split_rel > 0 && len > split_rel) { len = split_rel; }
    const bool is_last = finalize_at_end && len == T;
    nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                  static_cast<std::uint64_t>(len));
    const auto ws = workspaces();
    work_.reset();
    tp_->work->reset();
    // A chunk wider than the Program's arenas runs on each GPU's lent memory (T42).
    ArenaLoan loan0, loan1;
    const bool lent_arena = len > runtime[0]->experts->prompt_arena_columns();
    runtime[0]->experts->prepare_prompt(len);
    // One expert window per chunk (rank 0's backend drives both GPUs' slot tables).
    const auto abandon = [&] {
        for (int rank = kTensorParallelWidth - 1; rank >= 0; --rank) {
            runtime[static_cast<std::size_t>(rank)]->experts->abandon_window(rank_stream(rank));
        }
    };
    runtime[0]->experts->begin_window(s, Qwen4Window::Prompt);
    try {
        const ScopedCurrentDevice peer(execution.dev[1]->device);
        runtime[1]->experts->begin_window(rank_stream(1), Qwen4Window::Prompt);
    } catch (...) {
        runtime[0]->experts->abandon_window(s);
        throw;
    }
    try {
        if (lent_arena) {
            const cudaStream_t streams[2] = {s, rank_stream(1)};
            const auto spans = runtime[0]->experts->prompt_workspace(streams);
            loan0.lend(work_, spans[0]);
            loan1.lend(*tp_->work, spans[1]);
        }
        std::uint32_t ple_width = 0;
        if (config_.ple) {
            // One host gather of the chunk's n-gram rows, uploaded to both ranks below. The
            // staging is reused per chunk; the previous chunk ended in a synchronization of both
            // devices, so its uploads have completed.
            if (runtime[0]->ple_gather == nullptr || runtime[0]->ple_host == nullptr) {
                throw std::logic_error("Qwen4Exp prefill has no PLE gather staging");
            }
            ple_width = runtime[0]->ple_gather->width();
            const std::span<const std::int32_t> history(text_prefill->token_ids.data(),
                                                        text_prefill->token_ids.size());
            runtime[0]->ple_gather->gather(
                history, base, static_cast<std::size_t>(len),
                std::span<std::uint16_t>(runtime[0]->ple_host,
                                         static_cast<std::size_t>(ple_width) * len));
        }
        RankTensors ids_device, positions, rope_positions, ple_embedding, xf;
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            const ScopedCurrentDevice current(rank == 0 ? ctx_.device : execution.dev[1]->device);
            cudaStream_t stream = rank_stream(rank);
            ids_device[r]       = ws[r]->alloc(DType::I32, {len});
            CUDA_CHECK(cudaMemcpyAsync(ids_device[r].data, ids.data(), ids_device[r].bytes(),
                                       cudaMemcpyHostToDevice, stream));
            positions[r] = ws[r]->alloc(DType::I32, {len});
            ops::fill_i32_positions(positions[r], base_i, stream);
            rope_positions[r] = positions[r];
            if (rope_delta_ != 0) {
                rope_positions[r] = ws[r]->alloc(DType::I32, {len});
                ops::offset_i32_positions(positions[r], rank == 0 ? io_.rope_delta : tp_->rope_delta,
                                          rope_positions[r], stream);
            }
            if (config_.ple) {
                ple_embedding[r] =
                    ws[r]->alloc(DType::BF16, {static_cast<std::int32_t>(ple_width), len});
                CUDA_CHECK(cudaMemcpyAsync(ple_embedding[r].data, runtime[0]->ple_host,
                                           ple_embedding[r].bytes(), cudaMemcpyHostToDevice,
                                           stream));
            }
        }
        // The carried hidden of every prompt column: the committed wide residual. Rank 0 keeps
        // the whole chunk (prefill_hidden: causal scoring's staged columns); rank 1's copy feeds
        // the last column's vocabulary-split head. Under MTP rank 1 keeps its chunk in its own
        // prefill_hidden as well (bound only then): the Program's copy_tail retains rank 1's tail
        // hidden from there for the prefix-reuse bridge (T66b).
        const std::int32_t wide_rows = dimension(config_.residual_width());
        xf[0] = chunk_hidden(prefill_hidden_, work_, wide_rows, len);
        xf[1] = chunk_hidden(tp_->prefill_hidden, *tp_->work, wide_rows, len);
        std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            ranks[r]     = Qwen4Rank<WorkspaceArena>{
                    .config     = &*shard_config_,
                    .parameters = &rank_parameters(rank),
                    .work       = ws[r],
                    .stream     = rank_stream(rank),
                    .execution  = rank_execution_view(rank),
                    .device     = rank == 0 ? ctx_.device : execution.dev[1]->device,
                    .columns    = {.width            = len,
                                   .batch            = 1,
                                   .cache_positions  = &positions[r],
                                   .rope_positions   = &rope_positions[r],
                                   .valid_columns    = nullptr,
                                   .table_rows       = rank == 0 ? &io_.text_kv_table_row
                                                                 : &tp_->text_kv_table_row,
                                   .max_visible      = static_cast<std::uint32_t>(base_i + len),
                                   .prefill          = true,
                                   .source_slot      = linear_state_source_slot_,
                                   .destination_slot = linear_state_destination_slot_,
                                   .first_position   = base_i},
                    .state      = {.kv      = &rank_text_cache(rank),
                                   .linear  = &rank_state(rank),
                                   .runtime = runtime[r]}};
        }
        Qwen4Forward<WorkspaceArena> forward(config_, ranks, kTensorParallelWidth,
                                             {tp_->execution, tp_->events}, len, len);
        forward.run(ids_device, config_.ple ? &ple_embedding : nullptr, xf);

        if (is_last) {
            RankTensors last{xf[0].slice(1, len - 1, 1), xf[1].slice(1, len - 1, 1)};
            RankTensors x{work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1}),
                          tp_->work->alloc(DType::BF16, {dimension(config_.hidden_size), 1})};
            forward.head(last, x);
            Tensor logits = io_.logits.slice(1, 0, 1);
            logits_tp2(x, logits);
            // Sampling belongs to rank 0 alone, over the gathered complete logits.
            ops::set_i32_scalar(io_.pos, base_i + static_cast<int>(T), s);
            ops::set_i32_scalar(io_.rope_pos, base_i + static_cast<int>(T) + rope_delta_, s);
            if (sampling_config_ != nullptr) {
                ops::sample(logits, io_.token,
                            dimension(parameters_.model.resources().public_token_count),
                            sampling_config_, io_.pos, ops::kSamplePurposePrefill, work_, s);
            } else {
                ops::argmax(logits, io_.token,
                            dimension(parameters_.model.resources().public_token_count), s);
            }
        }
        if (mtp_enabled() && io_.mtp.has_value()) {
            // MTP alignment of this chunk on both ranks (as at one device).
            if (mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
                throw std::logic_error("Qwen4Exp MTP prefill exceeds the draft window");
            }
            const auto alignment_tokens =
                static_cast<std::uint32_t>(text_prefill->token_ids.size());
            const qwen3_5::MtpAlignmentWindow window = qwen3_5::plan_mtp_alignment_window(
                alignment_tokens, base, static_cast<std::uint32_t>(len));
            const int prompt_columns =
                len - static_cast<int>(window.final_column_uses_generated_token);
            auto mtp_scope0 = work_.scope();
            auto mtp_scope1 = tp_->work->scope();
            Tensor mtp_ids  = work_.alloc(DType::I32, {len});
            if (prompt_columns != 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    mtp_ids.data, text_prefill->token_ids.data() + window.shifted_embedding_begin,
                    static_cast<std::size_t>(prompt_columns) * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice, s));
            }
            if (window.final_column_uses_generated_token) {
                CUDA_CHECK(::ninfer::memcpy_device_async(mtp_ids.slice(0, len - 1, 1).data, io_.token.data,
                                           sizeof(std::int32_t), s));
            }
            qwen4_mtp_prefill_tp2(mtp_ids, xf, positions, rope_positions,
                                  static_cast<std::uint32_t>(base_i + len),
                                  is_last && mtp_proposal_extent_ != 0);
        }
        if (split_rel > 0 && len == split_rel && rewrite_checkpoint_hidden_output_ != nullptr) {
            const Tensor checkpoint = xf[0].slice(1, len - 1, 1);
            CUDA_CHECK(::ninfer::memcpy_device_async(rewrite_checkpoint_hidden_output_->data, checkpoint.data,
                                       checkpoint.bytes(), s));
            if (peer_rewrite_checkpoint_hidden_output_ != nullptr) {
                // Rank 1 retains the same column of its own replicated residual (T66b): the
                // MTP bridge resumes rank 1's half of the MTP head from it.
                const Tensor peer_checkpoint = xf[1].slice(1, len - 1, 1);
                require_shape(*peer_rewrite_checkpoint_hidden_output_, DType::BF16,
                              {wide_rows, 1}, "rank 1 rewrite checkpoint hidden output");
                const ScopedCurrentDevice peer(execution.dev[1]->device);
                CUDA_CHECK(::ninfer::memcpy_device_async(
                    peer_rewrite_checkpoint_hidden_output_->data, peer_checkpoint.data,
                    peer_checkpoint.bytes(), rank_stream(1)));
            }
        }
        keep_last_column(xf[0], prefill_hidden_, s);
        if (tp_->prefill_hidden.data != nullptr) {
            const ScopedCurrentDevice peer(execution.dev[1]->device);
            keep_last_column(xf[1], tp_->prefill_hidden, rank_stream(1));
        }
        // Observe, wait and close inside the try (T36 M2, as run_expert_window at one device): a
        // failure here must not leave either rank's window open for the next execution.
        prefill_split_frontier_ = -1;
        timing.begin_wait();
        runtime[0]->experts->observe_window(s);
        ctx_.synchronize();
        execution.dev[1]->synchronize();
        timing.end_wait();
        runtime[0]->experts->end_window(s);
        {
            const ScopedCurrentDevice peer(execution.dev[1]->device);
            runtime[1]->experts->end_window(rank_stream(1));
        }
    } catch (...) {
        try {
            ctx_.synchronize();
            execution.dev[1]->synchronize();
        } catch (...) {}
        abandon();
        throw;
    }
    loan1.restore();
    loan0.restore();
    work_.reset();
    tp_->work->reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(len),
                              .finalized        = finalize_at_end && len == T,
                              .timing           = timing.finish()};
}

void TextContext::qwen4_ordinary_decode_tp2(const Tensor& ids, const Tensor& cache_positions,
                                            const Tensor& rope_positions,
                                            const Tensor& kv_table_rows, const Tensor& source_slots,
                                            const Tensor& destination_slots,
                                            ops::CausalAttentionExecutionEnvelope envelope,
                                            Tensor& hidden, Tensor& logits) {
    if (tp_->ordinary == nullptr) {
        throw std::logic_error("tensor-parallel decode requires rank 1's ordinary frame");
    }
    const std::array<const Qwen4Runtime*, 2> runtime{&qwen4_runtime(), &qwen4_peer_runtime()};
    const std::int32_t batch = ids.ne[0];
    const auto peer_window   = [batch](const Tensor& source, const char* label) {
        if (source.dtype != DType::I32 || source.ne[0] < batch || source.ne[1] != 1 ||
            source.ne[2] != 1 || source.ne[3] != 1 || !source.is_contiguous() ||
            source.data == nullptr) {
            throw std::invalid_argument(std::string("tensor-parallel rank 1 ordinary ") + label +
                                        " does not cover the batch");
        }
        return source.slice(0, 0, batch);
    };
    const OrdinaryPeerFrame& frame = *tp_->ordinary;
    const RankTensors rank_ids{ids, peer_window(frame.tokens, "tokens")};
    const RankTensors cache{cache_positions, peer_window(frame.cache_positions, "cache positions")};
    const RankTensors rope{rope_positions, peer_window(frame.rope_positions, "RoPE positions")};
    const RankTensors rows{kv_table_rows, peer_window(frame.text_kv_table_rows, "KV rows")};
    const RankTensors sources{source_slots, peer_window(frame.state_source_slots, "source slots")};
    const RankTensors destinations{
        destination_slots, peer_window(frame.state_destination_slots, "destination slots")};
    RankTensors ple_rows;
    if (config_.ple) {
        for (std::size_t r = 0; r < 2; ++r) {
            const Tensor& all = runtime[r]->decode_ple_rows;
            if (all.data == nullptr || all.ne[1] < batch) {
                throw std::logic_error("Qwen4Exp decode has no PLE rows");
            }
            ple_rows[r] = all.slice(1, 0, batch);
        }
    }
    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    const auto ws = workspaces();
    work_.reset();
    tp_->work->reset();
    {
        std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            ranks[r]     = Qwen4Rank<WorkspaceArena>{
                    .config     = &*shard_config_,
                    .parameters = &rank_parameters(rank),
                    .work       = ws[r],
                    .stream     = rank_stream(rank),
                    .execution  = rank_execution_view(rank),
                    .device     = rank == 0 ? ctx_.device : execution.dev[1]->device,
                    .columns    = {.width             = 1,
                                   .batch             = batch,
                                   .cache_positions   = &cache[r],
                                   .rope_positions    = &rope[r],
                                   .valid_columns     = nullptr,
                                   .table_rows        = &rows[r],
                                   .max_visible       = envelope.max_visible_keys,
                                   .prefill           = false,
                                   .source_slots      = &sources[r],
                                   .destination_slots = &destinations[r]},
                    .state      = {.kv      = &rank_text_cache(rank),
                                   .linear  = &rank_state(rank),
                                   .runtime = runtime[r]}};
        }
        // Rank 0's carried wide residual is the call's hidden; rank 1's only feeds its
        // vocabulary half through the replicated output mixer.
        RankTensors wide{hidden,
                         ws[1]->alloc(DType::BF16, {dimension(config_.residual_width()), batch})};
        Qwen4Forward<WorkspaceArena> forward(config_, ranks, kTensorParallelWidth,
                                             {tp_->execution, tp_->events}, batch, batch);
        forward.run(rank_ids, config_.ple ? &ple_rows : nullptr, wide);
        RankTensors x{ws[0]->alloc(DType::BF16, {dimension(config_.hidden_size), batch}),
                      ws[1]->alloc(DType::BF16, {dimension(config_.hidden_size), batch})};
        forward.head(wide, x);
        logits_tp2(x, logits);
    }
    work_.reset();
    tp_->work->reset();
}


// ---- MTP at tp 2 (T33) -------------------------------------------------------------------------

Tensor TextContext::qwen4_ids_on_rank1(const Tensor& ids) {
    Tensor out = tp_->work->alloc(DType::I32, {static_cast<std::int32_t>(ids.numel())});
    const ops::PeerEvents& events = *tp_->events;
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), ctx_.stream));
    const ScopedCurrentDevice rank1(tp_->execution->dev[1]->device);
    CUDA_CHECK(cudaStreamWaitEvent(rank_stream(1), events.inputs_ready(0), 0));
    CUDA_CHECK(::ninfer::memcpy_device_async(out.data, ids.data, ids.bytes(),
                               rank_stream(1)));
    return out;
}

void TextContext::qwen4_target_verify_tp2(const RankTensors& ids, const RankTensors& cache_positions,
                                          const RankTensors& rope_positions,
                                          const RankTensors& valid_columns,
                                          const RankTensors& kv_table_rows,
                                          const RankTensors& source_slots,
                                          ops::CausalAttentionExecutionEnvelope envelope,
                                          const RankTensors& hidden, Tensor& logits,
                                          Tensor& target_tokens) {
    const std::array<const Qwen4Runtime*, 2> runtime{&qwen4_runtime(), &qwen4_peer_runtime()};
    const std::int32_t W = ids[0].ne[0];
    const std::int32_t B = ids[0].ne[1];
    const std::int32_t T = W * B;
    const std::int32_t wide_rows = dimension(config_.residual_width());
    const std::int32_t H         = dimension(config_.hidden_size);
    if (W < 2 || B <= 0 || B > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("Qwen4Exp verification window is outside [2,K+1]x[1,8]");
    }
    if (gdn_state_action_ != GdnStateAction::RecordForReplay || replay_records_ == nullptr) {
        throw std::logic_error("Qwen4Exp verification requires ReplaySSM records");
    }
    for (std::size_t r = 0; r < 2; ++r) {
        require_shape(ids[r], DType::I32, {W, B}, "tp 2 verify ids");
        require_shape(hidden[r], DType::BF16, {wide_rows, W, B}, "tp 2 verify hidden");
    }
    require_shape(logits, DType::BF16, {dimension(config_.vocab_size), W, B}, "verify logits");
    require_shape(target_tokens, DType::I32, {W, B}, "verify tokens");
    RankTensors rows;
    if (config_.ple) {
        for (std::size_t r = 0; r < 2; ++r) {
            if (runtime[r]->decode_ple_rows.data == nullptr || runtime[r]->decode_ple_rows.ne[1] < T) {
                throw std::logic_error("Qwen4Exp tp 2 verification has no PLE rows");
            }
            rows[r] = runtime[r]->decode_ple_rows.slice(1, 0, T);
        }
    }
    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    const auto ws = workspaces();
    work_.reset();
    tp_->work->reset();
    {
        std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            ranks[r]     = Qwen4Rank<WorkspaceArena>{
                    .config     = &*shard_config_,
                    .parameters = &rank_parameters(rank),
                    .work       = ws[r],
                    .stream     = rank_stream(rank),
                    .execution  = rank_execution_view(rank),
                    .device     = rank == 0 ? ctx_.device : execution.dev[1]->device,
                    .columns    = {.width             = W,
                                   .batch             = B,
                                   .cache_positions   = &cache_positions[r],
                                   .rope_positions    = &rope_positions[r],
                                   .valid_columns     = &valid_columns[r],
                                   .table_rows        = &kv_table_rows[r],
                                   .max_visible       = envelope.max_visible_keys,
                                   .prefill           = false,
                                   .source_slots      = &source_slots[r],
                                   .destination_slots = &source_slots[r],
                                   .replay_records    = &rank_replay_records(rank)},
                    .state      = {.kv      = &rank_text_cache(rank),
                                   .linear  = &rank_state(rank),
                                   .runtime = runtime[r]}};
        }
        Qwen4Forward<WorkspaceArena> forward(config_, ranks, kTensorParallelWidth,
                                             {tp_->execution, tp_->events}, T, T);
        const RankTensors flat_ids{ids[0].view({T}), ids[1].view({T})};
        RankTensors wide{hidden[0].view({wide_rows, T}), hidden[1].view({wide_rows, T})};
        forward.run(flat_ids, config_.ple ? &rows : nullptr, wide);
        RankTensors x{ws[0]->alloc(DType::BF16, {H, T}), ws[1]->alloc(DType::BF16, {H, T})};
        forward.head(wide, x);
        Tensor flat_logits = logits.view({dimension(config_.vocab_size), T});
        logits_tp2(x, flat_logits);
        Tensor flat_tokens = target_tokens.view({T});
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
    }
    work_.reset();
    tp_->work->reset();
}

void TextContext::qwen4_mtp_forward_decode_tp2(const Tensor& ids, const RankTensors& hidden,
                                               const RankTensors& cache_positions,
                                               const RankTensors& rope_positions,
                                               const RankTensors& valid_columns,
                                               const RankTensors& kv_table_rows,
                                               ops::CausalAttentionExecutionEnvelope envelope,
                                               const RankTensors& mtp_hidden) {
    const std::array<const Qwen4Runtime*, 2> runtime{&qwen4_runtime(), &qwen4_peer_runtime()};
    const Qwen4MtpParameters* peer_mtp =
        rank_parameters(1).qwen4_mtp ? &*rank_parameters(1).qwen4_mtp : nullptr;
    if (qwen4_mtp_ == nullptr || peer_mtp == nullptr) {
        throw std::logic_error("Qwen4Exp tp 2 MTP decode requires both ranks' MTP layers");
    }
    const std::int32_t W         = ids.ne[0];
    const std::int32_t B         = ids.ne[1];
    const std::int32_t T         = W * B;
    const std::int32_t wide_rows = dimension(config_.residual_width());
    require_shape(ids, DType::I32, {W, B}, "tp 2 MTP decode ids");
    for (std::size_t r = 0; r < 2; ++r) {
        require_shape(hidden[r], DType::BF16, {wide_rows, W, B}, "tp 2 MTP decode target hidden");
        require_shape(mtp_hidden[r], DType::BF16, {wide_rows, W, B}, "tp 2 MTP decode hidden");
    }
    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    const auto ws = workspaces();
    auto scope0   = work_.scope();
    auto scope1   = tp_->work->scope();
    nvtx::ScopedRange range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(T));
    const Tensor ids1 = qwen4_ids_on_rank1(ids);
    std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
    for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        ranks[r]     = Qwen4Rank<WorkspaceArena>{
                .config     = &*shard_config_,
                .parameters = &rank_parameters(rank),
                .work       = ws[r],
                .stream     = rank_stream(rank),
                .execution  = rank_execution_view(rank),
                .device     = rank == 0 ? ctx_.device : execution.dev[1]->device,
                .columns    = {.width               = W,
                               .batch               = B,
                               .cache_positions     = &cache_positions[r],
                               .rope_positions      = &rope_positions[r],
                               .valid_columns       = &valid_columns[r],
                               .moe_last_valid_only = true, // as qwen4_mtp_forward_decode
                               .table_rows          = &kv_table_rows[r],
                               .max_visible         = envelope.max_visible_keys,
                               .prefill             = false},
                .state      = {.kv      = &rank_mtp_cache(rank),
                               .linear  = &rank_state(rank),
                               .runtime = runtime[r]}};
    }
    Qwen4Forward<WorkspaceArena> forward(config_, ranks, kTensorParallelWidth,
                                         {tp_->execution, tp_->events}, T, T);
    const RankTensors flat_ids{ids.view({T}), ids1.view({T})};
    const RankTensors in{hidden[0].view({wide_rows, T}), hidden[1].view({wide_rows, T})};
    RankTensors out{mtp_hidden[0].view({wide_rows, T}), mtp_hidden[1].view({wide_rows, T})};
    forward.run_mtp({qwen4_mtp_, peer_mtp}, flat_ids, nullptr, in, out, false);
}

void TextContext::qwen4_mtp_propose_tp2(const RankTensors& hidden, Tensor& logits,
                                        Tensor& draft_tokens, Tensor* draft_logprobs) {
    const Qwen4MtpParameters* peer_mtp =
        rank_parameters(1).qwen4_mtp ? &*rank_parameters(1).qwen4_mtp : nullptr;
    if (qwen4_mtp_ == nullptr || peer_mtp == nullptr) {
        throw std::logic_error("Qwen4Exp tp 2 proposal requires both ranks' MTP layers");
    }
    const std::int32_t B         = hidden[0].ne[1];
    const std::int32_t wide_rows = dimension(config_.residual_width());
    const std::int32_t H         = dimension(config_.hidden_size);
    for (std::size_t r = 0; r < 2; ++r) {
        require_shape(hidden[r], DType::BF16, {wide_rows, B}, "tp 2 MTP proposal hidden");
    }
    require_shape(draft_tokens, DType::I32, {B}, "MTP proposal tokens");
    if (proposal_head_ != nullptr) { throw std::logic_error("Qwen4Exp has no trimmed proposal head yet"); }
    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    const auto ws = workspaces();
    auto scope0   = work_.scope();
    auto scope1   = tp_->work->scope();
    nvtx::ScopedRange range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(B));
    std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
    for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        ranks[r]     = Qwen4Rank<WorkspaceArena>{
                .config     = &*shard_config_,
                .parameters = &rank_parameters(rank),
                .work       = ws[r],
                .stream     = rank_stream(rank),
                .execution  = rank_execution_view(rank),
                .device     = rank == 0 ? ctx_.device : execution.dev[1]->device,
                .columns    = {.width = B, .batch = 1, .prefill = false},
                .state      = {.kv = nullptr, .linear = &rank_state(rank), .runtime = nullptr}};
    }
    Qwen4Forward<WorkspaceArena> forward(config_, ranks, kTensorParallelWidth,
                                         {tp_->execution, tp_->events}, B, B);
    RankTensors x{ws[0]->alloc(DType::BF16, {H, B}), ws[1]->alloc(DType::BF16, {H, B})};
    forward.head({&qwen4_mtp_->head, &peer_mtp->head}, hidden, x);
    Tensor window = logits.slice(1, 0, B);
    // T43: the shortlist head runs on rank 0 alone (its prefix is a view of rank 0's vocabulary
    // half, its extra rows were copied there at startup); no logits leave rank 1. The next MTP
    // step reads the drafts on rank 1 through qwen4_ids_on_rank1, as with the split head.
    if (qwen4_runtime_ != nullptr && qwen4_runtime_->draft_head != nullptr) {
        const Qwen4DraftHead& draft = *qwen4_runtime_->draft_head;
        if (!draft.split) {
            draft_head_propose(draft, x[0], window, draft_tokens, draft_logprobs, work_,
                               ctx_.stream);
            return;
        }
        // Split shortlist: each rank scores its half, one exchange of the packed candidates,
        // rank 0 selects (lower row on ties) and maps the row to its token, per column.
        if (draft_logprobs != nullptr) {
            throw std::invalid_argument(
                "the split draft shortlist has no log-probabilities (--mtp-min-p at tp 2: set "
                "NINFER_QWEN4_DRAFT_SPLIT=0)");
        }
        for (std::int32_t b = 0; b < B; ++b) {
            auto column0 = ws[0]->scope();
            auto column1 = ws[1]->scope();
            const std::array<workspace::TensorParallelProposalRoots, 2> roots{
                workspace::tp_proposal_argmax(*ws[0], draft.prefix_rows, 1),
                workspace::tp_proposal_argmax(*ws[1], draft.peer_rows, 1)};
            const RankTensors column{x[0].slice(1, b, 1), x[1].slice(1, b, 1)};
            Tensor token = draft_tokens.slice(0, b, 1);
            proposal_argmax_split(column, {&draft.prefix, &draft.peer},
                                  {roots[0].partial, roots[1].partial},
                                  {roots[0].local, roots[1].local},
                                  {roots[0].candidates, roots[1].candidates},
                                  {roots[0].staging, roots[1].staging}, token, ws,
                                  *tp_->execution, *tp_->events);
        }
        ops::proposal_remap_token_ids(draft_tokens, draft.token_ids, draft.rows(), ctx_.stream);
        return;
    }
    logits_tp2(x, window);
    const std::int32_t vocabulary = dimension(parameters_.model.resources().public_token_count);
    ops::argmax(window, draft_tokens, vocabulary, ctx_.stream);
    if (draft_logprobs != nullptr) {
        ops::target_logprobs(window, draft_tokens, vocabulary, *draft_logprobs, ctx_.stream);
    }
}

// The MTP KV of one prompt chunk at tp 2 (as qwen4_mtp_prefill): every column appended on both
// ranks; with `propose` the last column runs the layer and drafts mtp_proposal_extent_ tokens.
void TextContext::qwen4_mtp_prefill_tp2(const Tensor& ids, const RankTensors& hidden,
                                        const RankTensors& positions,
                                        const RankTensors& rope_positions,
                                        std::uint32_t max_visible, bool propose) {
    const std::array<const Qwen4Runtime*, 2> runtime{&qwen4_runtime(), &qwen4_peer_runtime()};
    const Qwen4MtpParameters* peer_mtp =
        rank_parameters(1).qwen4_mtp ? &*rank_parameters(1).qwen4_mtp : nullptr;
    if (qwen4_mtp_ == nullptr || peer_mtp == nullptr || tp_->mtp == nullptr) {
        throw std::logic_error("Qwen4Exp tp 2 MTP prefill requires both ranks' MTP layers");
    }
    if (runtime[0]->taps != nullptr && runtime[0]->taps->wants_mtp_logits()) {
        throw std::invalid_argument("Qwen4Exp MTP teacher forcing runs at one device");
    }
    const ExecutionContext& execution = *tp_->execution;
    const auto ws                     = workspaces();
    const std::int32_t T              = ids.ne[0];
    const std::int32_t append         = propose ? T - 1 : T;
    const std::int32_t wide_rows      = dimension(config_.residual_width());
    nvtx::ScopedRange range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(T));
    const auto run = [&](const RankTensors& column_ids, const RankTensors& in, RankTensors& out,
                         const RankTensors& pos, const RankTensors& rope, std::int32_t count,
                         bool append_only, std::uint32_t visible_keys) {
        std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            ranks[r]     = Qwen4Rank<WorkspaceArena>{
                    .config     = &*shard_config_,
                    .parameters = &rank_parameters(rank),
                    .work       = ws[r],
                    .stream     = rank_stream(rank),
                    .execution  = rank_execution_view(rank),
                    .device     = rank == 0 ? ctx_.device : execution.dev[1]->device,
                    .columns    = {.width           = count,
                                   .batch           = 1,
                                   .cache_positions = &pos[r],
                                   .rope_positions  = &rope[r],
                                   .valid_columns   = nullptr,
                                   .table_rows      = rank == 0 ? &io_.backend_kv_table_row
                                                                : &tp_->backend_kv_table_row,
                                   .max_visible     = visible_keys,
                                   .prefill         = true},
                    .state      = {.kv      = &rank_mtp_cache(rank),
                                   .linear  = &rank_state(rank),
                                   .runtime = runtime[r]}};
        }
        Qwen4Forward<WorkspaceArena> forward(config_, ranks, kTensorParallelWidth,
                                             {tp_->execution, tp_->events}, count, count);
        forward.run_mtp({qwen4_mtp_, peer_mtp}, column_ids, nullptr, in, out, append_only);
    };
    const Tensor ids1 = qwen4_ids_on_rank1(ids);
    if (append > 0) {
        auto scope0 = work_.scope();
        auto scope1 = tp_->work->scope();
        RankTensors scratch{ws[0]->alloc(DType::BF16, {wide_rows, append}),
                            ws[1]->alloc(DType::BF16, {wide_rows, append})};
        run({ids.slice(0, 0, append), ids1.slice(0, 0, append)},
            {hidden[0].slice(1, 0, append), hidden[1].slice(1, 0, append)}, scratch,
            {positions[0].slice(0, 0, append), positions[1].slice(0, 0, append)},
            {rope_positions[0].slice(0, 0, append), rope_positions[1].slice(0, 0, append)}, append,
            true, max_visible);
    }
    if (!propose) { return; }
    RankTensors ar_hidden{io_.mtp->ar_hidden, tp_->mtp->ar_hidden};
    run({ids.slice(0, T - 1, 1), ids1.slice(0, T - 1, 1)},
        {hidden[0].slice(1, T - 1, 1), hidden[1].slice(1, T - 1, 1)}, ar_hidden,
        {positions[0].slice(0, T - 1, 1), positions[1].slice(0, T - 1, 1)},
        {rope_positions[0].slice(0, T - 1, 1), rope_positions[1].slice(0, T - 1, 1)}, 1, false,
        max_visible);
    Tensor logits = io_.logits.slice(1, 0, 1);
    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
    qwen4_mtp_propose_tp2(ar_hidden, logits, draft0);
    // Autoregressive drafts: draft i at position (frontier + i) from draft i-1 and the previous
    // MTP hidden, each rank on its own position copy.
    const RankTensors ar_position{io_.mtp->position.slice(0, 0, 1), tp_->mtp->position.slice(0, 0, 1)};
    for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
        const ScopedCurrentDevice current(rank == 0 ? ctx_.device : execution.dev[1]->device);
        Tensor p = ar_position[static_cast<std::size_t>(rank)];
        ops::set_i32_scalar(p, static_cast<std::int32_t>(max_visible), rank_stream(rank));
    }
    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
        auto scope0 = work_.scope();
        auto scope1 = tp_->work->scope();
        const Tensor previous = io_.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next           = io_.mtp->draft_tokens.slice(0, i, 1);
        const Tensor previous1 = qwen4_ids_on_rank1(previous);
        RankTensors rope, next_hidden;
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            const ScopedCurrentDevice current(rank == 0 ? ctx_.device : execution.dev[1]->device);
            rope[r]        = ws[r]->alloc(DType::I32, {1});
            next_hidden[r] = ws[r]->alloc(DType::BF16, {wide_rows, 1});
            ops::offset_i32_positions(ar_position[r], rank == 0 ? io_.rope_delta : tp_->rope_delta,
                                      rope[r], rank_stream(rank));
        }
        const auto visible = max_visible + static_cast<std::uint32_t>(i);
        run({previous, previous1}, ar_hidden, next_hidden, ar_position, rope, 1, false, visible);
        qwen4_mtp_propose_tp2(next_hidden, logits, next);
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            const ScopedCurrentDevice current(rank == 0 ? ctx_.device : execution.dev[1]->device);
            CUDA_CHECK(::ninfer::memcpy_device_async(ar_hidden[r].data, next_hidden[r].data, ar_hidden[r].bytes(), rank_stream(rank)));
            Tensor p = ar_position[r];
            ops::increment_i32_scalar(p, rank_stream(rank));
        }
    }
}

// T66b: one sequence's MTP forward over T consecutive positions at tp 2 (the prefix-reuse bridge
// column and its autoregressive proposal steps), as qwen4_mtp_forward_batch at one device: each
// rank appends its MTP QSA heads' K/V on its own prefill MTP KV row and runs its expert partition
// from its own copy of the target hidden; with `logits_column` >= 0 rank 0 proposes from that
// column through qwen4_mtp_propose_tp2. `rope_positions` are each rank's [T] or [T,3] positions
// (the Text RoPE of this architecture is 1-D: axis 0).
void TextContext::qwen4_mtp_forward_batch_tp2(const Tensor& ids, const RankTensors& hidden,
                                              const RankTensors& positions,
                                              const RankTensors& rope_positions,
                                              ops::CausalAttentionExecutionEnvelope envelope,
                                              const RankTensors& mtp_hidden, int logits_column,
                                              Tensor* logits, Tensor* draft_token,
                                              const Tensor* input_embeddings) {
    const std::array<const Qwen4Runtime*, 2> runtime{&qwen4_runtime(), &qwen4_peer_runtime()};
    const Qwen4MtpParameters* peer_mtp =
        rank_parameters(1).qwen4_mtp ? &*rank_parameters(1).qwen4_mtp : nullptr;
    if (qwen4_mtp_ == nullptr || peer_mtp == nullptr || batch_mtp_kv_ == nullptr ||
        !batch_mtp_kv_->has_qsa_index() || tp_->mtp == nullptr) {
        throw std::logic_error("Qwen4Exp tp 2 MTP forward requires both ranks' MTP layers");
    }
    if (input_embeddings != nullptr) {
        throw std::invalid_argument("Qwen4Exp MTP takes token embeddings only (no image input)");
    }
    const std::int32_t T         = ids.ne[0];
    const std::int32_t wide_rows = dimension(config_.residual_width());
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_ || logits_column >= T) {
        throw std::invalid_argument("Qwen4Exp MTP batch columns are outside [1,prefill_chunk]");
    }
    require_shape(ids, DType::I32, {T}, "tp 2 MTP ids");
    RankTensors rope;
    for (std::size_t r = 0; r < 2; ++r) {
        require_shape(positions[r], DType::I32, {T}, "tp 2 MTP positions");
        require_shape(hidden[r], DType::BF16, {wide_rows, T}, "tp 2 MTP hidden");
        require_shape(mtp_hidden[r], DType::BF16, {wide_rows, T}, "tp 2 MTP output hidden");
        const Tensor& given = rope_positions[r];
        if (given.dtype != DType::I32 || given.ne[0] != T || (given.ne[1] != 1 && given.ne[1] != 3) ||
            given.ne[2] != 1 || given.ne[3] != 1 || !given.is_contiguous() || given.data == nullptr) {
            throw std::invalid_argument("tp 2 MTP RoPE positions must be [T] or [T,3]");
        }
        rope[r] = given.ne[1] == 3 ? given.slice(1, 0, 1).view({T}) : given.view({T});
    }
    if (logits_column >= 0 && (logits == nullptr || draft_token == nullptr)) {
        throw std::invalid_argument("Qwen4Exp MTP logits and draft outputs are required");
    }
    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    const auto ws = workspaces();
    auto scope0   = work_.scope();
    auto scope1   = tp_->work->scope();
    nvtx::ScopedRange range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(T));
    const Tensor ids1 = qwen4_ids_on_rank1(ids);
    {
        std::array<Qwen4Rank<WorkspaceArena>, 2> ranks;
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            ranks[r]     = Qwen4Rank<WorkspaceArena>{
                    .config     = &*shard_config_,
                    .parameters = &rank_parameters(rank),
                    .work       = ws[r],
                    .stream     = rank_stream(rank),
                    .execution  = rank_execution_view(rank),
                    .device     = rank == 0 ? ctx_.device : execution.dev[1]->device,
                    .columns    = {.width           = T,
                                   .batch           = 1,
                                   .cache_positions = &positions[r],
                                   .rope_positions  = &rope[r],
                                   .valid_columns   = nullptr,
                                   .table_rows      = rank == 0 ? &io_.backend_kv_table_row
                                                                : &tp_->backend_kv_table_row,
                                   .max_visible     = envelope.max_visible_keys,
                                   .prefill         = true},
                    .state      = {.kv      = &rank_mtp_cache(rank),
                                   .linear  = &rank_state(rank),
                                   .runtime = runtime[r]}};
        }
        Qwen4Forward<WorkspaceArena> forward(config_, ranks, kTensorParallelWidth,
                                             {tp_->execution, tp_->events}, T, T);
        RankTensors out = mtp_hidden;
        forward.run_mtp({qwen4_mtp_, peer_mtp}, {ids, ids1}, nullptr, hidden, out, false);
    }
    if (logits_column >= 0) {
        const RankTensors column{mtp_hidden[0].slice(1, logits_column, 1),
                                 mtp_hidden[1].slice(1, logits_column, 1)};
        qwen4_mtp_propose_tp2(column, *logits, *draft_token);
    }
}

} // namespace ninfer::models::qwen3_5::execution
