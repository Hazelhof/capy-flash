#include "ninfer/ops/copy_rows.h"

#include "core/device.h"

#include <cstdint>

namespace ninfer::ops {
namespace {

template <class Word>
__global__ void copy_rows_kernel(Word* __restrict__ destination, const Word* __restrict__ source,
                                 std::size_t source_pitch_words, std::size_t row_words,
                                 std::size_t columns) {
    const std::size_t total  = row_words * columns;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const std::size_t column = i / row_words;
        const std::size_t word   = i - column * row_words;
        destination[i]           = source[column * source_pitch_words + word];
    }
}

template <class Word>
void launch(void* destination, const void* source, std::size_t source_pitch, std::size_t row_bytes,
            std::size_t columns, cudaStream_t stream) {
    const std::size_t w     = sizeof(Word);
    const std::size_t total = row_bytes / w * columns;
    const std::size_t want  = (total + 255) / 256;
    const int blocks        = static_cast<int>(want < 1 ? 1 : (want > 1024 ? 1024 : want));
    copy_rows_kernel<Word><<<blocks, 256, 0, stream>>>(static_cast<Word*>(destination),
                                                       static_cast<const Word*>(source),
                                                       source_pitch / w, row_bytes / w, columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void copy_rows(void* destination, const void* source, std::size_t source_pitch, std::size_t row_bytes,
               std::size_t columns, cudaStream_t stream) {
    if (row_bytes == 0 || columns == 0) { return; }
    const auto aligned = [&](std::size_t a) {
        return ((reinterpret_cast<std::uintptr_t>(destination) | reinterpret_cast<std::uintptr_t>(source) |
                 source_pitch | row_bytes) % a) == 0;
    };
    if (aligned(16)) {
        launch<uint4>(destination, source, source_pitch, row_bytes, columns, stream);
    } else if (aligned(4)) {
        launch<std::uint32_t>(destination, source, source_pitch, row_bytes, columns, stream);
    } else {
        launch<std::uint8_t>(destination, source, source_pitch, row_bytes, columns, stream);
    }
}

} // namespace ninfer::ops
