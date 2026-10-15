#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// Existing dense Op ABI uses Weight axes directly; callers establish BF16/FP32 and layout.
inline Tensor as_dense(const Weight& weight) {
    const auto dtype = weight.qtype == QType::FP32 ? DType::FP32 : DType::BF16;
    auto* data       = const_cast<void*>(weight.qdata);
    switch (weight.ndim) {
    case 1:
        return Tensor(data, dtype, {weight.shape[0]});
    case 2:
        return Tensor(data, dtype, {weight.shape[0], weight.shape[1]});
    case 3:
        return Tensor(data, dtype, {weight.shape[0], weight.shape[1], weight.shape[2]});
    default:
        return Tensor(data, dtype,
                      {weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]});
    }
}

// Geometry describes one complete encoded parent. Regions address its logical elements;
// padding and scale words do not belong to that element domain.
struct WeightGeometry {
    QType format       = QType::BF16;
    QuantLayout layout = QuantLayout::Contiguous;
    std::vector<std::uint64_t> shape;
    std::uint64_t elements            = 0;
    std::uint64_t bytes               = 0;
    std::uint64_t alignment           = 256;
    std::uint64_t padded_columns      = 0;
    std::uint64_t group_size          = 0;
    std::uint64_t code_bytes_per_row  = 0;
    std::uint64_t high_bytes_per_row  = 0;
    std::uint64_t scale_bytes_per_row = 0;
    std::uint64_t code_bytes          = 0;
    std::uint64_t high_offset         = 0;
    std::uint64_t high_bytes          = 0;
    std::uint64_t scale_offset        = 0;
    std::uint64_t scale_bytes         = 0;
    std::uint64_t divisor_offset      = 0;
    // NVFP4 FP32 weight divisors after the scale plane. A stacked plane (separately quantised
    // source matrices of equal row counts, e.g. a layer's routed experts) keeps one per source:
    // row r uses word r / (rows / divisor_count). Each source covers whole 128-row scale tiles.
    std::uint64_t divisor_count = 1;

    [[nodiscard]] std::uint64_t rows_per_divisor() const noexcept {
        return shape.empty() ? 0 : shape.front() / divisor_count;
    }
};

[[nodiscard]] WeightGeometry weight_geometry(QType format, QuantLayout layout,
                                             std::span<const std::uint64_t> shape,
                                             std::uint64_t divisors = 1);
[[nodiscard]] std::uint64_t weight_element_count(std::span<const std::uint64_t> shape);

struct WeightParent {
    WeightGeometry geometry;
    const std::byte* data = nullptr;
    // The NVFP4 divisor of a single-divisor parent; 0 for other formats and for a stacked plane,
    // whose per-source words are at geometry.divisor_offset.
    float weight_scale_divisor = 0.0F;
};

// Byte ranges, from the first encoded byte of a parent, that hold whole logical rows
// [row_begin, row_begin + row_count): their codes (and RowSplit high bits), their scales and, for a
// stacked NVFP4 plane, the divisor words of the sources they belong to. Empty ranges are omitted
// and adjacent ones joined. BlockScaleK16M128x4 needs 128-row aligned rows (its scales are
// swizzled in 128-row tiles stored tile-row-major) and a stacked plane whole sources; a
// single-divisor NVFP4 parent's one word is not part of any row block and is not returned.
struct ByteSpan {
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;

    friend bool operator==(const ByteSpan&, const ByteSpan&) = default;
};
[[nodiscard]] std::vector<ByteSpan> weight_row_block_spans(const WeightGeometry& geometry,
                                                           std::uint64_t row_begin,
                                                           std::uint64_t row_count);

struct WeightRegion {
    const WeightParent* parent = nullptr;
    std::uint64_t begin        = 0;
    std::uint64_t end          = 0;
};

struct WeightView {
    std::vector<std::uint64_t> shape;
    std::vector<WeightRegion> parts;
};

// Coalesce adjacent logical regions without moving their bytes. Fails for a gather or
// multiple parents; those remain explicit inputs for a consumer supporting that form.
[[nodiscard]] WeightRegion contiguous_weight_region(const WeightView& view);
[[nodiscard]] bool is_complete_weight(const WeightView& view);
[[nodiscard]] std::uint64_t weight_scale_offset(const WeightGeometry& geometry, std::uint64_t row,
                                                std::uint64_t group);

struct WeightRowPlanes {
    const std::byte* codes        = nullptr;
    const std::byte* high         = nullptr;
    const std::byte* scales       = nullptr;
    std::uint64_t row_begin       = 0;
    std::uint64_t row_count       = 0;
    std::uint64_t code_row_bytes  = 0;
    std::uint64_t high_row_bytes  = 0;
    std::uint64_t scale_row_bytes = 0;
    // NVFP4 keeps the parent scale base; use its row origin and block geometry.
    bool swizzled_scales = false;
};

[[nodiscard]] WeightRowPlanes weight_row_planes(const WeightRegion& region);
[[nodiscard]] Tensor weight_tensor(const WeightView& view,
                                   std::initializer_list<std::int32_t> internal_shape);
// Existing Weight ABI: complete quantized parents, direct regions, and RowSplit row views.
// Arbitrary FP8/NVFP4 regions use their explicit planes until a native consumer supports them.
[[nodiscard]] Weight native_weight(const WeightView& view, float input_divisor = 0.0F);

} // namespace ninfer
