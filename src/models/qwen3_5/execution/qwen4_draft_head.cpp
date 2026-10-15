#include "models/qwen3_5/execution/qwen4_draft_head.h"

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {

DraftShortlistSplit split_draft_shortlist(std::span<const std::int32_t> ids,
                                          std::int32_t vocabulary, std::int32_t prefix_limit,
                                          std::int32_t multiple) {
    if (ids.empty()) { throw std::invalid_argument("draft shortlist is empty"); }
    if (vocabulary <= 0 || prefix_limit < 0 || multiple <= 0) {
        throw std::invalid_argument("draft shortlist: invalid vocabulary or prefix limit");
    }
    std::vector<std::int32_t> sorted(ids.begin(), ids.end());
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 || sorted.back() >= vocabulary) {
        throw std::invalid_argument("draft shortlist: token ID outside [0, " +
                                    std::to_string(vocabulary) + ")");
    }
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument("draft shortlist: duplicate token ID");
    }
    // Round the shortlist up to a multiple of `multiple` rows with the lowest IDs it lacks.
    if (const auto rest = sorted.size() % static_cast<std::size_t>(multiple); rest != 0) {
        std::vector<std::int32_t> added;
        std::size_t at = 0;
        for (std::int32_t id = 0; id < vocabulary && added.size() + rest < std::size_t(multiple);
             ++id) {
            while (at < sorted.size() && sorted[at] < id) { ++at; }
            if (at == sorted.size() || sorted[at] != id) { added.push_back(id); }
        }
        sorted.insert(sorted.end(), added.begin(), added.end());
        std::sort(sorted.begin(), sorted.end());
    }
    DraftShortlistSplit out;
    while (out.prefix < static_cast<std::int32_t>(sorted.size()) && out.prefix < prefix_limit &&
           sorted[static_cast<std::size_t>(out.prefix)] == out.prefix) {
        ++out.prefix;
    }
    out.prefix -= out.prefix % multiple; // the extra block starts aligned in the logits
    out.extras.assign(sorted.begin() + out.prefix, sorted.end());
    return out;
}

std::vector<std::int32_t> read_draft_shortlist(const char* path, std::uint32_t rows) {
    std::FILE* file = std::fopen(path, "r");
    if (file == nullptr) {
        throw std::runtime_error(std::string("cannot open the draft shortlist ") + path);
    }
    std::vector<std::int32_t> ids;
    long long value = 0;
    bool number = false, comment = false;
    int c = 0;
    const auto flush = [&] {
        if (number) {
            if (value > std::numeric_limits<std::int32_t>::max()) {
                std::fclose(file);
                throw std::invalid_argument("draft shortlist: token ID overflows int32");
            }
            ids.push_back(static_cast<std::int32_t>(value));
        }
        number = false;
        value  = 0;
    };
    while ((c = std::fgetc(file)) != EOF) {
        if (comment) {
            if (c == '\n') { comment = false; }
            continue;
        }
        if (c == '#') {
            flush();
            comment = true;
        } else if (c >= '0' && c <= '9') {
            value  = value * 10 + (c - '0');
            number = true;
        } else if (c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == ',') {
            flush();
        } else {
            std::fclose(file);
            throw std::invalid_argument("draft shortlist: unexpected character in " +
                                        std::string(path));
        }
    }
    flush();
    std::fclose(file);
    if (rows > 0 && ids.size() > rows) { ids.resize(rows); }
    if (ids.empty()) { throw std::invalid_argument("draft shortlist: no token IDs in the file"); }
    return ids;
}

bool draft_head_supported(const Weight& head) noexcept {
    return head.qtype == QType::Q6_G64_FP16 && head.layout == QuantLayout::RowSplit &&
           head.qdata != nullptr && head.qhigh != nullptr && head.scales != nullptr &&
           head.n > 0 && head.k > 0 && head.padded_shape[1] >= head.k &&
           head.padded_shape[1] % 128 == 0;
}

std::array<std::size_t, 3> q6_row_plane_bytes(const Weight& head) {
    if (!draft_head_supported(head)) {
        throw std::invalid_argument("draft head: the output head is not a Q6 RowSplit weight");
    }
    const auto groups = static_cast<std::size_t>(head.padded_shape[1] / 64);
    return {groups * 32, groups * 16, groups * 2};
}

LinearParameters q6_row_view(const LinearParameters& head, std::int32_t first, std::int32_t rows) {
    const auto bytes = q6_row_plane_bytes(head.weight);
    if (first < 0 || rows <= 0 || first > head.weight.n - rows) {
        throw std::invalid_argument("draft head: row view outside the output head");
    }
    LinearParameters out = head;
    Weight& w            = out.weight;
    const auto f         = static_cast<std::size_t>(first);
    w.qdata  = static_cast<const std::byte*>(head.weight.qdata) + f * bytes[0];
    w.qhigh  = static_cast<const std::byte*>(head.weight.qhigh) + f * bytes[1];
    w.scales = static_cast<const std::byte*>(head.weight.scales) + f * bytes[2];
    w.n               = rows;
    w.shape[0]        = rows;
    w.padded_shape[0] = rows;
    w.payload         = w.qdata;
    w.payload_bytes   = 0; // a view: the planes are not one payload
    return out;
}

LinearParameters q6_rows_weight(const LinearParameters& head,
                                const std::array<std::byte*, 3>& planes, std::int32_t rows) {
    (void)q6_row_plane_bytes(head.weight);
    if (rows <= 0 || planes[0] == nullptr || planes[1] == nullptr || planes[2] == nullptr) {
        throw std::invalid_argument("draft head: gathered rows need planes");
    }
    LinearParameters out = head;
    Weight& w            = out.weight;
    w.qdata           = planes[0];
    w.qhigh           = planes[1];
    w.scales          = planes[2];
    w.n               = rows;
    w.shape[0]        = rows;
    w.padded_shape[0] = rows;
    w.payload         = planes[0];
    w.payload_bytes   = 0;
    return out;
}

void gather_q6_rows(const Weight& head, std::span<const std::int32_t> source_rows,
                    const std::array<std::byte*, 3>& planes, std::int32_t destination_row0,
                    cudaStream_t stream) {
    const auto bytes = q6_row_plane_bytes(head);
    const std::array<const std::byte*, 3> source{static_cast<const std::byte*>(head.qdata),
                                                 static_cast<const std::byte*>(head.qhigh),
                                                 static_cast<const std::byte*>(head.scales)};
    // Consecutive source rows go as one copy per plane.
    std::size_t i = 0;
    while (i < source_rows.size()) {
        const std::int32_t first = source_rows[i];
        if (first < 0 || first >= head.n) {
            throw std::invalid_argument("draft head: gathered row outside the output head");
        }
        std::size_t run = 1;
        while (i + run < source_rows.size() &&
               source_rows[i + run] == first + static_cast<std::int32_t>(run) &&
               first + static_cast<std::int32_t>(run) < head.n) {
            ++run;
        }
        const auto destination = static_cast<std::size_t>(destination_row0) + i;
        for (std::size_t p = 0; p < 3; ++p) {
            CUDA_CHECK(::ninfer::memcpy_device_async(planes[p] + destination * bytes[p],
                                       source[p] + static_cast<std::size_t>(first) * bytes[p],
                                       run * bytes[p], stream));
        }
        i += run;
    }
}

void draft_head_propose(const Qwen4DraftHead& head, const Tensor& x, const Tensor& logits,
                        Tensor& tokens, Tensor* logprobs, WorkspaceArena& work,
                        cudaStream_t stream) {
    const std::int32_t rows = head.rows();
    const std::int32_t B    = x.ne[1];
    if (rows <= 0 || head.token_ids == nullptr) {
        throw std::logic_error("draft head has no shortlist");
    }
    if (logits.dtype != DType::BF16 || !logits.is_contiguous() ||
        static_cast<std::int64_t>(logits.numel()) < static_cast<std::int64_t>(rows) * B) {
        throw std::invalid_argument("draft head: the proposal logits buffer is too small");
    }
    if (tokens.dtype != DType::I32 || tokens.numel() != B || !tokens.is_contiguous()) {
        throw std::invalid_argument("draft head: proposal tokens must be I32 [B]");
    }
    Tensor window(logits.data, DType::BF16, {rows, B});
    for (std::int32_t b = 0; b < B; ++b) {
        const Tensor column = B == 1 ? x : x.slice(1, b, 1);
        auto* base = static_cast<std::uint16_t*>(logits.data) + static_cast<std::size_t>(b) * rows;
        if (head.prefix_rows > 0) {
            Tensor out(base, DType::BF16, {head.prefix_rows, 1});
            ops::linear(column, head.prefix.weight, out, head.prefix.policy, work, stream);
        }
        if (head.extra_rows > 0) {
            Tensor out(base + head.prefix_rows, DType::BF16, {head.extra_rows, 1});
            ops::linear(column, head.extra.weight, out, head.extra.policy, work, stream);
        }
    }
    ops::argmax(window, tokens, rows, stream);
    if (logprobs != nullptr) { ops::target_logprobs(window, tokens, rows, *logprobs, stream); }
    ops::proposal_remap_token_ids(tokens, head.token_ids, rows, stream);
}

} // namespace ninfer::models::qwen3_5::execution
