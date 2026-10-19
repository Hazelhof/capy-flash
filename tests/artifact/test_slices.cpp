// Byte-exact tensor slicing for every registered layout. Parents and expected shards are both
// written from logical coordinates: the shard side evaluates the same coordinate pattern at the
// parent coordinate each shard element came from. The two meet only at the logical level, so a
// misplaced plane, row tile, group or swizzled scale shows up as a byte mismatch.

#include "artifact/schema.h"
#include "artifact/slices.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;

using Bytes = std::vector<std::uint8_t>;
using Map   = std::function<std::uint64_t(std::uint64_t)>;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::uint8_t pattern(std::uint64_t row, std::uint64_t column, std::uint64_t byte) {
    return static_cast<std::uint8_t>((row * 131 + column * 17 + byte * 7 + 11) & 0xFF);
}

std::uint64_t word_bytes(QType format) { return format == QType::BF16 ? 2 : 4; }

// Encodes a tensor of `shape` whose element (n, k) carries the pattern of parent element
// (row_of(n), column_of(k)). Plane offsets come from weight_geometry; the NVFP4 scale swizzle is
// written from the layout formula.
Bytes build(QType format, QuantLayout layout, const std::vector<std::uint64_t>& shape,
            const Map& row_of, const Map& column_of) {
    const auto g = weight_geometry(format, layout, shape);
    Bytes out(static_cast<std::size_t>(g.bytes), 0);
    const auto rows    = shape[0];
    const auto columns = shape.size() > 1 ? weight_element_count(std::span(shape).subspan(1)) : 1;
    for (std::uint64_t n = 0; n < rows; ++n) {
        const auto row = row_of(n);
        if (layout == QuantLayout::Contiguous) {
            const auto word = word_bytes(format);
            for (std::uint64_t k = 0; k < columns; ++k) {
                for (std::uint64_t i = 0; i < word; ++i) {
                    out[(n * columns + k) * word + i] = pattern(row, column_of(k), i);
                }
            }
        } else if (layout == QuantLayout::RowSplit) {
            const auto groups = g.padded_columns / g.group_size;
            const auto code   = g.code_bytes_per_row / groups;
            const auto high   = g.high_bytes_per_row / groups;
            for (std::uint64_t group = 0; group < groups; ++group) {
                const auto source = column_of(group * g.group_size) / g.group_size;
                const auto index  = n * groups + group;
                for (std::uint64_t i = 0; i < code; ++i) {
                    out[index * code + i] = pattern(row, source, i);
                }
                for (std::uint64_t i = 0; i < high; ++i) {
                    out[g.high_offset + index * high + i] = pattern(row, source, 100 + i);
                }
                for (std::uint64_t i = 0; i < 2; ++i) {
                    out[g.scale_offset + index * 2 + i] = pattern(row, source, 200 + i);
                }
            }
        } else if (layout == QuantLayout::BlockScaleK32) {
            // MXFP8 from the format definition: N x K codes, then at the next multiple of 256
            // bytes N x K/32 scale words, both row-major.
            const auto scale_plane = (rows * columns + 255) / 256 * 256;
            for (std::uint64_t k = 0; k < columns; ++k) {
                out[n * columns + k] = pattern(row, column_of(k), 0);
            }
            for (std::uint64_t group = 0; group < columns / 32; ++group) {
                out[scale_plane + n * (columns / 32) + group] =
                    pattern(row, column_of(group * 32) / 32, 200);
            }
        } else if (layout == QuantLayout::RowInterleaved) {
            // Row-scaled NVFP4 from the format definition: rows of K/2 code bytes then K/16
            // scale bytes, nothing else.
            const auto stride = columns / 2 + columns / 16;
            for (std::uint64_t j = 0; j < columns / 2; ++j) {
                out[n * stride + j] = pattern(row, column_of(2 * j), 0);
            }
            for (std::uint64_t group = 0; group < columns / 16; ++group) {
                out[n * stride + columns / 2 + group] =
                    pattern(row, column_of(group * 16) / 16, 200);
            }
        } else if (layout == QuantLayout::RowScale) {
            for (std::uint64_t k = 0; k < columns; ++k) {
                out[n * columns + k] = pattern(row, column_of(k), 0);
            }
            for (std::uint64_t i = 0; i < 2; ++i) {
                out[g.scale_offset + n * 2 + i] = pattern(row, 0, 200 + i);
            }
        } else {
            for (std::uint64_t j = 0; j < columns / 2; ++j) {
                out[n * (columns / 2) + j] = pattern(row, column_of(2 * j), 0);
            }
            for (std::uint64_t group = 0; group < columns / 16; ++group) {
                const auto inner  = n % 128;
                const auto offset = (n / 128 * (columns / 64) + group / 4) * 512 + inner % 32 * 16 +
                                    inner / 32 * 4 + group % 4;
                out[g.scale_offset + offset] = pattern(row, column_of(group * 16) / 16, 200);
            }
        }
    }
    if (format == QType::NVFP4) {
        for (std::uint64_t i = 0; i < 4; ++i) {
            out[g.divisor_offset + i] = static_cast<std::uint8_t>(0x30 + i);
        }
    }
    return out;
}

Bytes apply(const Bytes& parent, const TensorSlice& slice) {
    Bytes out(static_cast<std::size_t>(slice.geometry.bytes), 0);
    for (const auto& copy : slice.copies) {
        if (copy.source_offset + copy.bytes > parent.size() ||
            copy.dest_offset + copy.bytes > out.size()) {
            check(false, "slice copy escapes its parent or shard");
            return out;
        }
        std::copy_n(parent.begin() + static_cast<std::ptrdiff_t>(copy.source_offset), copy.bytes,
                    out.begin() + static_cast<std::ptrdiff_t>(copy.dest_offset));
    }
    return out;
}

Map identity() {
    return [](std::uint64_t value) { return value; };
}

// Shard coordinate -> parent coordinate for ranges concatenated in order.
Map through(std::vector<SliceRange> ranges) {
    return [ranges = std::move(ranges)](std::uint64_t value) {
        for (const auto& range : ranges) {
            if (value < range.count) { return range.begin + value; }
            value -= range.count;
        }
        throw std::out_of_range("coordinate outside the selected ranges");
    };
}

std::uint64_t total(const std::vector<SliceRange>& ranges) {
    std::uint64_t out = 0;
    for (const auto& range : ranges) { out += range.count; }
    return out;
}

void expect_rows(QType format, QuantLayout layout, const std::vector<std::uint64_t>& shape,
                 const std::vector<SliceRange>& rows, const std::string& label) {
    const auto parent_geometry = weight_geometry(format, layout, shape);
    const auto parent          = build(format, layout, shape, identity(), identity());
    const auto slice           = tensor_row_slice(parent_geometry, rows);
    auto shard_shape           = shape;
    shard_shape[0]             = total(rows);
    check(slice.geometry.shape == shard_shape &&
              slice.geometry.bytes == weight_geometry(format, layout, shard_shape).bytes,
          label + ": shard geometry");
    check(apply(parent, slice) == build(format, layout, shard_shape, through(rows), identity()),
          label + ": shard bytes");
}

void expect_columns(QType format, QuantLayout layout, const std::vector<std::uint64_t>& shape,
                    const std::vector<SliceRange>& columns, const std::string& label) {
    const auto parent_geometry = weight_geometry(format, layout, shape);
    const auto parent          = build(format, layout, shape, identity(), identity());
    const auto slice           = tensor_column_slice(parent_geometry, columns);
    const std::vector<std::uint64_t> shard_shape{shape[0], total(columns)};
    check(slice.geometry.shape == shard_shape &&
              slice.geometry.bytes == weight_geometry(format, layout, shard_shape).bytes,
          label + ": shard geometry");
    check(apply(parent, slice) == build(format, layout, shard_shape, identity(), through(columns)),
          label + ": shard bytes");
}

template <class Fn>
void expect_rejected(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const ArtifactError&) { return; }
    check(false, label + ": accepted");
}

void contiguous() {
    const auto g =
        weight_geometry(QType::BF16, QuantLayout::Contiguous, std::vector<std::uint64_t>{8, 6});
    expect_rows(QType::BF16, QuantLayout::Contiguous, {8, 6}, {{0, 2}, {5, 3}}, "bf16 rows");
    expect_rows(QType::FP32, QuantLayout::Contiguous, {10}, {{3, 4}}, "fp32 vector rows");
    expect_rows(QType::BF16, QuantLayout::Contiguous, {4, 3, 2}, {{1, 2}}, "bf16 rank-3 rows");
    expect_columns(QType::BF16, QuantLayout::Contiguous, {8, 6}, {{2, 3}}, "bf16 columns");
    expect_columns(QType::INT32, QuantLayout::Contiguous, {8, 6}, {{0, 1}, {2, 2}, {5, 1}},
                   "int32 multi-range columns");
    const auto whole = tensor_column_slice(g, std::vector<SliceRange>{{0, 6}});
    check(whole.copies.size() == 1 && whole.copies.front().bytes == g.bytes,
          "complete-width column slice coalesces into one copy");
    const auto rows = [&](std::vector<SliceRange> ranges) {
        return [&g, ranges] { (void)tensor_row_slice(g, ranges); };
    };
    const auto columns = [&](std::vector<SliceRange> ranges) {
        return [&g, ranges] { (void)tensor_column_slice(g, ranges); };
    };
    expect_rejected(rows({}), "empty row range list");
    expect_rejected(rows({{2, 0}}), "zero-length row range");
    expect_rejected(rows({{4, 4}, {0, 2}}), "descending row ranges");
    expect_rejected(rows({{0, 5}, {4, 2}}), "overlapping row ranges");
    expect_rejected(rows({{6, 4}}), "row range past the last row");
    expect_rejected(columns({{2, 2}, {1, 1}}), "descending column ranges");
    expect_rejected(columns({{0, 3}, {2, 2}}), "overlapping column ranges");
    expect_rejected(columns({{4, 3}}), "column range past the last column");
    const auto vector =
        weight_geometry(QType::FP32, QuantLayout::Contiguous, std::vector<std::uint64_t>{10});
    expect_rejected([&] { (void)tensor_column_slice(vector, std::vector<SliceRange>{{0, 2}}); },
                    "column slice of a vector");
    expect_rejected(
        [&] { (void)tensor_slice(g, ShardAxis::Replicated, std::vector<SliceRange>{{0, 2}}); },
        "slice for a complete-parent placement");
}

void row_split() {
    for (const auto format :
         {QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16, QType::Q8_G32_FP16}) {
        const auto label = "row-split format " + std::to_string(static_cast<int>(format));
        expect_rows(format, QuantLayout::RowSplit, {6, 256}, {{1, 2}, {4, 2}}, label + " rows");
        expect_columns(format, QuantLayout::RowSplit, {6, 384}, {{128, 256}}, label + " columns");
        // K=200 pads to 256: a leading 128-column shard needs none of the padding groups, and
        // the unaligned remainder cannot form a shard of this layout.
        expect_columns(format, QuantLayout::RowSplit, {4, 200}, {{0, 128}},
                       label + " ragged-K leading columns");
        const auto g =
            weight_geometry(format, QuantLayout::RowSplit, std::vector<std::uint64_t>{4, 200});
        expect_rejected([&] { (void)tensor_column_slice(g, std::vector<SliceRange>{{128, 72}}); },
                        label + " ragged-K trailing columns");
        expect_rejected([&] { (void)tensor_column_slice(g, std::vector<SliceRange>{{64, 64}}); },
                        label + " columns off the 128 boundary");
        const auto wide =
            weight_geometry(format, QuantLayout::RowSplit, std::vector<std::uint64_t>{4, 384});
        expect_rejected(
            [&] { (void)tensor_column_slice(wide, std::vector<SliceRange>{{0, 128}, {256, 128}}); },
            label + " multi-range columns");
    }
}

void block_scale() {
    // Three 128-row tiles and three 64-column scale tiles.
    expect_rows(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, {384, 192}, {{0, 128}, {256, 128}},
                "nvfp4 rows");
    expect_columns(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, {384, 192}, {{64, 128}},
                   "nvfp4 columns");
    const auto g = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                   std::vector<std::uint64_t>{384, 192});
    expect_rejected([&] { (void)tensor_row_slice(g, std::vector<SliceRange>{{64, 128}}); },
                    "nvfp4 rows off the 128-row tile");
    expect_rejected([&] { (void)tensor_column_slice(g, std::vector<SliceRange>{{32, 64}}); },
                    "nvfp4 columns off the 64-column tile");
    expect_rejected(
        [&] { (void)tensor_column_slice(g, std::vector<SliceRange>{{0, 64}, {128, 64}}); },
        "nvfp4 multi-range columns");
}

void row_scale() {
    expect_rows(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale, {6, 40}, {{2, 3}}, "fp8 rows");
    expect_rows(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale, {6, 40}, {{0, 1}, {3, 3}},
                "fp8 multi-range rows");
    // Every row keeps its multiplier, so the whole scale plane is replicated.
    expect_columns(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale, {6, 40}, {{8, 17}},
                   "fp8 columns");
    const auto g = weight_geometry(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale,
                                   std::vector<std::uint64_t>{6, 40});
    expect_rejected([&] { (void)tensor_column_slice(g, std::vector<SliceRange>{{0, 8}, {16, 8}}); },
                    "fp8 multi-range columns");
}

template <class Fn>
void expect_invalid(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return; }
    check(false, label + ": accepted");
}

// MXFP8 (block_scale_k32_v1): sizes written out from the format definition, both TP2 halves of
// either axis, row blocks and scale coordinates. N = 48 and 96 are real shapes that are not
// multiples of 128 (the GDN a|b projection and its TP2 half).
void block_scale_k32() {
    constexpr auto mx  = QType::MXFP8_E4M3FN_G32_UE8M0;
    constexpr auto k32 = QuantLayout::BlockScaleK32;
    const auto g       = weight_geometry(mx, k32, std::vector<std::uint64_t>{96, 256});
    check(g.group_size == 32 && g.code_bytes == 96 * 256 && g.scale_offset == 96 * 256 &&
              g.scale_bytes == 96 * 8 && g.bytes == 96 * 256 + 96 * 8 && g.divisor_offset == 0,
          "mxfp8 geometry");
    const auto small = weight_geometry(mx, k32, std::vector<std::uint64_t>{3, 64});
    check(small.scale_offset == 256 && small.bytes == 256 + 6, "mxfp8 scale plane alignment");
    expect_rows(mx, k32, {96, 256}, {{0, 48}}, "mxfp8 rows, first TP2 half");
    expect_rows(mx, k32, {96, 256}, {{48, 48}}, "mxfp8 rows, second TP2 half");
    expect_rows(mx, k32, {96, 256}, {{1, 7}, {40, 13}}, "mxfp8 multi-range rows");
    expect_columns(mx, k32, {6, 256}, {{0, 128}}, "mxfp8 columns, first TP2 half");
    expect_columns(mx, k32, {6, 256}, {{128, 128}}, "mxfp8 columns, second TP2 half");
    expect_columns(mx, k32, {5, 96}, {{32, 32}}, "mxfp8 middle column group");
    expect_rejected([&] { (void)tensor_column_slice(g, std::vector<SliceRange>{{16, 32}}); },
                    "mxfp8 columns off the 32-column group");
    expect_rejected(
        [&] { (void)tensor_column_slice(g, std::vector<SliceRange>{{0, 32}, {64, 32}}); },
        "mxfp8 multi-range columns");
    expect_invalid([&] { (void)weight_geometry(mx, k32, std::vector<std::uint64_t>{4, 48}); },
                   "mxfp8 K not a multiple of 32");
    expect_invalid(
        [&] { (void)weight_geometry(QType::NVFP4, k32, std::vector<std::uint64_t>{128, 64}); },
        "BlockScaleK32 with another format");
    expect_invalid(
        [&] {
            (void)weight_geometry(mx, QuantLayout::RowScale, std::vector<std::uint64_t>{4, 64});
        },
        "mxfp8 in another layout");
    const auto spans = weight_row_block_spans(g, 10, 5);
    check(spans.size() == 2 && spans[0] == ByteSpan{10 * 256, 5 * 256} &&
              spans[1] == ByteSpan{96 * 256 + 10 * 8, 5 * 8},
          "mxfp8 row block spans");
    check(weight_scale_offset(g, 7, 3) == 96 * 256 + 7 * 8 + 3, "mxfp8 scale coordinate");
}

// Row-scaled NVFP4 (row_interleaved_v1): the n-gram table row of 160 values is 90 bytes.
void row_interleaved() {
    constexpr auto fp4  = QType::NVFP4_ROW_E4M3_G16;
    constexpr auto rows = QuantLayout::RowInterleaved;
    const auto g        = weight_geometry(fp4, rows, std::vector<std::uint64_t>{10, 160});
    check(g.group_size == 16 && g.code_bytes_per_row == 90 && g.scale_offset == 80 &&
              g.scale_bytes_per_row == 10 && g.scale_bytes == 0 && g.code_bytes == 900 &&
              g.bytes == 900 && g.divisor_offset == 0,
          "row-scaled nvfp4 geometry");
    expect_rows(fp4, rows, {10, 160}, {{0, 5}}, "nvfp4 rows, first TP2 half");
    expect_rows(fp4, rows, {10, 160}, {{5, 5}}, "nvfp4 rows, second TP2 half");
    expect_rows(fp4, rows, {10, 160}, {{1, 2}, {6, 3}}, "nvfp4 multi-range rows");
    expect_columns(fp4, rows, {4, 160}, {{0, 80}}, "nvfp4 row columns, first half");
    expect_columns(fp4, rows, {4, 160}, {{80, 80}}, "nvfp4 row columns, second half");
    expect_columns(fp4, rows, {3, 64}, {{16, 32}}, "nvfp4 row middle groups");
    expect_rejected([&] { (void)tensor_column_slice(g, std::vector<SliceRange>{{8, 16}}); },
                    "nvfp4 row columns off the 16-column group");
    expect_invalid([&] { (void)weight_geometry(fp4, rows, std::vector<std::uint64_t>{4, 40}); },
                   "nvfp4 row K not a multiple of 16");
    expect_invalid(
        [&] {
            (void)weight_geometry(QType::FP8_E4M3FN_ROW_BF16, rows,
                                  std::vector<std::uint64_t>{4, 64});
        },
        "RowInterleaved with another format");
    const auto spans = weight_row_block_spans(g, 3, 4);
    check(spans.size() == 1 && spans[0] == ByteSpan{270, 360}, "nvfp4 row block span");
    check(weight_scale_offset(g, 2, 9) == 2 * 90 + 80 + 9, "nvfp4 row scale coordinate");
}

} // namespace

// A stacked NVFP4 plane (one divisor per source matrix, e.g. a layer's routed experts) and the
// byte spans of a row block. Oracle: weight_scale_offset() of every (row, group) of the block, and
// the plane sizes written out by hand.
void stacked_row_blocks() {
    const std::vector<std::uint64_t> shape{1024, 128};
    const auto g = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, shape, 8);
    check(g.divisor_count == 8 && g.rows_per_divisor() == 128, "stacked divisor count");
    check(g.bytes == g.divisor_offset + 8 * 4 && g.divisor_offset == g.scale_offset + 1024 * 8,
          "stacked plane size");
    const auto single = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, shape);
    check(single.bytes + 7 * 4 == g.bytes, "stacked plane adds one word per extra source");
    bool rejected = false;
    try {
        (void)weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, shape, 16);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "a divisor covering 64 rows (half a scale tile) was accepted");
    rejected = false;
    try {
        (void)weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit, shape, 2);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "a grouped integer format accepted several divisors");
    // Rows 256..511: sources 2 and 3 (an expert's gate and up in a gate|up bank).
    const auto spans = weight_row_block_spans(g, 256, 256);
    check(spans.size() == 3, "stacked row block is codes, scales and divisors");
    if (spans.size() == 3) {
        check(spans[0] == ByteSpan{256 * 64, 256 * 64}, "code span");
        std::uint64_t low = ~0ULL, high = 0, count = 0;
        for (std::uint64_t row = 256; row < 512; ++row) {
            for (std::uint64_t group = 0; group < 8; ++group) {
                const auto offset = weight_scale_offset(g, row, group);
                low               = std::min(low, offset);
                high              = std::max(high, offset);
                ++count;
            }
        }
        check(spans[1] == ByteSpan{low, high - low + 1} && count == high - low + 1,
              "scale span is not the contiguous set of the block's swizzled scales");
        check(spans[2] == ByteSpan{g.divisor_offset + 2 * 4, 2 * 4}, "divisor span");
    }
    const auto down = weight_row_block_spans(single, 128, 128);
    check(down.size() == 2, "a single-divisor plane's word is not part of a row block");
    for (const auto [begin, count] : {std::pair{64ULL, 128ULL}, std::pair{128ULL, 64ULL}}) {
        rejected = false;
        try {
            (void)weight_row_block_spans(g, begin, count);
        } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "a row block off the 128-row tiles was accepted");
    }
    rejected = false;
    try {
        const std::vector<SliceRange> rows{{0, 512}};
        (void)tensor_slice(g, ShardAxis::Rows, rows);
    } catch (const ArtifactError&) { rejected = true; }
    check(rejected, "a stacked plane was sliced");
    const auto q8   = weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit,
                                      std::vector<std::uint64_t>{4, 200});
    const auto rows = weight_row_block_spans(q8, 1, 2);
    check(rows.size() == 2 &&
              rows[0] == ByteSpan{q8.code_bytes_per_row, 2 * q8.code_bytes_per_row} &&
              rows[1] ==
                  ByteSpan{q8.scale_offset + q8.scale_bytes_per_row, 2 * q8.scale_bytes_per_row},
          "RowSplit row block spans");
}

int main() {
    try {
        contiguous();
        row_split();
        block_scale();
        row_scale();
        block_scale_k32();
        row_interleaved();
        stacked_row_blocks();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures) {
        std::cerr << failures << " tensor slice check(s) failed\n";
        return 1;
    }
    std::cout << "tensor slice checks passed\n";
    return 0;
}
