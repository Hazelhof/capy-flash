// Qwen4Exp n-gram (PLE) host gather.
//  1. Hash: the 16 table rows of every column equal the independent Python reference
//     (ninfer-ext tests/models/qwen4_exp/reference.py ngram_row_ids) at the published constants,
//     including sequence start, EOS-closed windows and repeated EOS (exact oracle).
//  2. Residency: mapped (with and without WILLNEED prefetch) and streamed (O_DIRECT) gathers give
//     the same BF16 embedding over a table split across two volume files at unaligned offsets, so
//     rows straddle 4 KiB pages; mapped decode equals E4M3 x BF16 multiplier rounded once to BF16.
// Part 2 ports ninfer-ext tests/models/qwen3_5/test_ple_gather.cpp (Apache-2.0).
#include "models/qwen3_5/execution/ple_gather.h"

#include <sys/mman.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace ninfer::models::qwen3_5::execution;

namespace {

#include "ple_hash_expected.inc"

constexpr std::uint32_t kOfficialEos = 248044;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

PleHashConfig official_hash() {
    PleHashConfig hash;
    hash.ngram_size      = 3;
    hash.heads_per_ngram = 8;
    hash.eos_token_id    = kOfficialEos;
    hash.multipliers     = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
    hash.head_moduli     = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
                            20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
                            20000153, 20000159, 20000161, 20000171};
    std::uint64_t offset = 0;
    for (const auto modulus : hash.head_moduli) {
        hash.head_offsets.push_back(offset);
        offset += modulus;
    }
    return hash;
}

int hash_case() {
    // A one-row placeholder mapping: the hash never reads the table.
    static const std::byte placeholder[16]{};
    PleTableView table;
    table.rows        = 320001536ULL;
    table.row_width   = 160;
    table.row_bytes   = 160;
    table.scale_plane = table.rows * 160;
    table.segments    = {{0, sizeof(placeholder), placeholder, {}, 0}};
    PleGather gather(table, official_hash(), PleResidency::Mapped);
    const std::vector<std::int32_t> tokens(std::begin(kHashTokens), std::end(kHashTokens));
    int failures = 0;
    std::vector<std::uint64_t> rows(16);
    for (std::size_t p = 0; p < tokens.size(); ++p) {
        gather.column_rows(tokens, p, rows);
        for (std::size_t h = 0; h < 16; ++h) {
            if (rows[h] != kHashRows[p][h]) {
                std::cerr << "n-gram row mismatch at column " << p << " head " << h << ": "
                          << rows[h] << " != " << kHashRows[p][h] << '\n';
                ++failures;
                break;
            }
        }
    }
    return failures;
}

constexpr std::uint32_t kRows     = 5000;
constexpr std::uint32_t kRowWidth = 16;
constexpr std::uint32_t kEos      = 7;

void write_file(const std::filesystem::path& path, std::uint64_t leading, const std::byte* data,
                std::size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::vector<char> pad(leading, 'x');
    out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    // Trailing bytes so the last direct page read is not short.
    const std::vector<char> tail(8192, 'y');
    out.write(tail.data(), static_cast<std::streamsize>(tail.size()));
}

struct Mapping {
    void* base        = MAP_FAILED;
    std::size_t bytes = 0;
    Mapping(const std::filesystem::path& path) {
        bytes  = std::filesystem::file_size(path);
        FILE* f = std::fopen(path.c_str(), "rb");
        base    = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fileno(f), 0);
        std::fclose(f);
        if (base == MAP_FAILED) throw std::runtime_error("mmap failed");
    }
    ~Mapping() {
        if (base != MAP_FAILED) ::munmap(base, bytes);
    }
    const std::byte* at(std::uint64_t offset) const {
        return static_cast<const std::byte*>(base) + offset;
    }
};

int residency_case(std::uint64_t split_bytes_extra = 0) {
    // Two 2-gram heads and two 3-gram heads of 16 columns: a 64-wide embedding.
    PleHashConfig hash;
    hash.ngram_size      = 3;
    hash.heads_per_ngram = 2;
    hash.eos_token_id    = kEos;
    hash.multipliers     = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
    hash.head_moduli     = {1201, 1213, 1217, 1223};
    std::uint64_t offset = 0;
    for (const auto modulus : hash.head_moduli) {
        hash.head_offsets.push_back(offset);
        offset += modulus;
    }

    // Object: FP8 codes [rows, 16], then BF16 row multipliers at a 256-byte aligned plane.
    const std::uint64_t scale_plane = (kRows * kRowWidth + 255) / 256 * 256;
    std::vector<std::byte> object(scale_plane + kRows * 2);
    std::mt19937 rng(1234);
    for (std::uint64_t i = 0; i < kRows * kRowWidth; ++i) {
        std::uint8_t code = static_cast<std::uint8_t>(rng());
        if ((code & 0x7F) == 0x7F) { code ^= 1; } // finite E4M3FN codes only
        object[i] = std::byte{code};
    }
    for (std::uint32_t r = 0; r < kRows; ++r) {
        const std::uint16_t scale = static_cast<std::uint16_t>(0x3C00 + (rng() % 0x0300));
        std::memcpy(object.data() + scale_plane + r * 2, &scale, 2);
    }
    // split_bytes_extra != 0 puts the volume boundary inside row 1777 (as a real artifact whose
    // part files end at 4 KiB-aligned payloads does): that row's codes straddle two segments.
    const std::uint64_t split = 1777 * kRowWidth + split_bytes_extra;
    // Direct I/O needs a filesystem that supports O_DIRECT: use the working directory.
    const auto dir = std::filesystem::current_path() / "ple_gather_test_volumes";
    std::filesystem::create_directories(dir);
    write_file(dir / "a.bin", 1234, object.data(), split);
    write_file(dir / "b.bin", 777, object.data() + split, object.size() - split);
    int failures = 0;
    {
        // The mapped segments address real file mappings, as the loader's mapping does.
        const Mapping a(dir / "a.bin"), b(dir / "b.bin");
        PleTableView table;
        table.rows        = kRows;
        table.row_width   = kRowWidth;
        table.row_bytes   = kRowWidth;
        table.scale_plane = scale_plane;
        table.segments    = {
            {0, split, a.at(1234), dir / "a.bin", 1234},
            {split, object.size() - split, b.at(777), dir / "b.bin", 777},
        };
        std::vector<std::int32_t> tokens(300);
        for (auto& token : tokens) { token = static_cast<std::int32_t>(rng() % 50000); }
        tokens[5] = tokens[140] = static_cast<std::int32_t>(kEos);
        if (split_bytes_extra != 0) {
            // Plant a 3-token context at positions 200..202 whose rows include the straddling row.
            PleGather probe(table, hash, PleResidency::Mapped, false);
            const std::uint64_t target = split / kRowWidth;
            std::vector<std::uint64_t> r(hash.heads());
            bool found = false;
            for (std::uint32_t attempt = 0; attempt < 2000000 && !found; ++attempt) {
                const std::int64_t c[3] = {static_cast<std::int64_t>(rng() % 50000),
                                           static_cast<std::int64_t>(rng() % 50000),
                                           static_cast<std::int64_t>(rng() % 50000)};
                probe.rows(c, r);
                for (const auto row : r) {
                    if (row == target) {
                        tokens[202] = static_cast<std::int32_t>(c[0]);
                        tokens[201] = static_cast<std::int32_t>(c[1]);
                        tokens[200] = static_cast<std::int32_t>(c[2]);
                        found       = true;
                    }
                }
            }
            failures += check(found, "no context reaches the straddling row");
        }

        PleGather mapped(table, hash, PleResidency::Mapped);
        PleGather plain(table, hash, PleResidency::Mapped, false);
        std::unique_ptr<PleGather> streamed;
        try {
            streamed = std::make_unique<PleGather>(table, hash, PleResidency::Stream);
        } catch (const std::runtime_error& error) {
            std::cerr << "note: direct I/O unavailable here (" << error.what()
                      << "); stream residency not compared\n";
        }
        for (const auto [first, count] :
             {std::pair<std::size_t, std::size_t>{0, 1}, {6, 3}, {141, 1}, {10, 290}}) {
            std::vector<std::uint16_t> a_out(count * mapped.width()), b_out(a_out.size()),
                c_out(a_out.size());
            mapped.gather(tokens, first, count, a_out);
            plain.gather(tokens, first, count, c_out);
            failures += check(a_out == c_out, "prefetched mapped rows differ from plain mapped");
            // T53: an earlier prefetch() of the same columns (and a no-op one on plain) is a hint.
            std::vector<std::uint16_t> d_out(a_out.size());
            mapped.prefetch(tokens, first, count);
            plain.prefetch(tokens, first, count);
            mapped.gather(tokens, first, count, d_out);
            failures += check(a_out == d_out, "rows gathered after prefetch() differ");
            if (streamed) {
                streamed->gather(tokens, first, count, b_out);
                failures += check(a_out == b_out, "streamed n-gram rows differ from mapped rows");
            }
        }
        // Mapped decode is independently the E4M3 row times its BF16 multiplier.
        std::vector<std::uint16_t> column(mapped.width());
        const std::size_t probe_column = split_bytes_extra != 0 ? 202 : 20;
        mapped.gather(tokens, probe_column, 1, column);
        std::vector<std::uint64_t> rows(hash.heads());
        const std::int64_t context[3] = {tokens[probe_column], tokens[probe_column - 1],
                                         tokens[probe_column - 2]};
        mapped.rows(context, rows);
        const auto e4m3 = [](std::uint8_t c) {
            const int e = (c >> 3) & 15, m = c & 7;
            const float v = e == 0 ? std::ldexp(m / 8.0F, -6) : std::ldexp(1.0F + m / 8.0F, e - 7);
            return (c & 0x80) ? -v : v;
        };
        for (std::uint32_t h = 0; h < hash.heads(); ++h) {
            std::uint16_t scale_bits;
            std::memcpy(&scale_bits, object.data() + scale_plane + rows[h] * 2, 2);
            float scale;
            const std::uint32_t scale_word = static_cast<std::uint32_t>(scale_bits) << 16;
            std::memcpy(&scale, &scale_word, 4);
            for (std::uint32_t i = 0; i < kRowWidth; ++i) {
                const float value =
                    e4m3(static_cast<std::uint8_t>(object[rows[h] * kRowWidth + i])) * scale;
                std::uint32_t bits;
                std::memcpy(&bits, &value, 4);
                const auto bf16 =
                    static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
                if (column[h * kRowWidth + i] != bf16) {
                    std::cerr << "n-gram row decode mismatch at head " << h << '\n';
                    ++failures;
                    break;
                }
            }
        }
    }
    std::filesystem::remove_all(dir);
    return failures;
}

} // namespace

int main() {
    try {
        int failures = hash_case();
        failures += residency_case();
        failures += residency_case(5); // a row straddling two volume segments (T155)
        std::cout << (failures == 0 ? "OK" : "FAIL") << " PLE gather\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "PLE gather: " << error.what() << '\n';
        return 1;
    }
}
