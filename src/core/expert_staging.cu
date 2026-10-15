#include "core/expert_staging.h"

#include "core/device.h" // CUDA_CHECK

#include <cuda.h>

#include <stdexcept>
#include <string>

namespace ninfer::expert_cache {
namespace {

constexpr int kThreads = kStagingExperts;

__global__ void __launch_bounds__(kThreads)
    staging_overlay_kernel(const std::int32_t* __restrict__ table, const std::uint32_t* tags,
                           int slots, int first_slot, const std::uint32_t* epoch, int layer,
                           const std::int32_t* __restrict__ ids, int routes,
                           std::int32_t* __restrict__ out, std::uint32_t* stats) {
    __shared__ std::int32_t row[kStagingExperts];
    __shared__ std::int32_t mark[kStagingExperts];
    __shared__ unsigned landed, used, used_routes;
    const int tid = static_cast<int>(threadIdx.x);
    row[tid]      = table[tid];
    mark[tid]     = 0;
    if (tid == 0) {
        landed      = 0;
        used        = 0;
        used_routes = 0;
    }
    // The epoch and the tags change between replays (and tags during this one): read them as
    // they are now, once.
    const std::uint32_t current = *reinterpret_cast<const volatile std::uint32_t*>(epoch);
    __syncthreads();
    for (int s = tid; s < slots; s += kThreads) {
        const std::uint32_t tag = reinterpret_cast<const volatile std::uint32_t*>(tags)[s];
        if (tag == kStagingTagEmpty || (tag >> 16) != current ||
            static_cast<int>((tag >> 9) & 0x7FU) != layer) {
            continue;
        }
        const int e = static_cast<int>(tag & 0x1FFU);
        atomicAdd(&landed, 1U);
        // A resident expert keeps its slot; one expert staged twice in one window cannot happen
        // (one batch per layer and window, distinct candidates): the higher slot would win.
        if (row[e] < 0) { atomicMax(&row[e], first_slot + s); }
    }
    __syncthreads();
    for (int i = tid; i < routes; i += kThreads) {
        const int e = ids[i];
        if (e < 0 || e >= kStagingExperts || table[e] >= 0 || row[e] < 0) { continue; }
        atomicAdd(&used_routes, 1U);
        if (atomicExch(&mark[e], 1) == 0) { atomicAdd(&used, 1U); }
    }
    out[tid] = row[tid];
    __syncthreads();
    if (tid == 0 && stats != nullptr) {
        stats[1] = landed;
        stats[2] = used;
        stats[3] = used_routes;
        __threadfence_system();
        stats[0] = current;
    }
}

} // namespace

void staging_overlay(const std::int32_t* table, const std::uint32_t* tags, std::int32_t slots,
                     std::int32_t first_slot, const std::uint32_t* epoch, std::int32_t layer,
                     const std::int32_t* ids, std::int32_t routes, std::int32_t* out,
                     std::uint32_t* stats, cudaStream_t stream) {
    if (table == nullptr || tags == nullptr || epoch == nullptr || out == nullptr ||
        (routes > 0 && ids == nullptr)) {
        throw std::invalid_argument("staging_overlay: null pointer");
    }
    if (slots < 0 || slots > kMaxStagingSlots || first_slot < 0 || layer < 0 ||
        layer >= kMaxStagingLayers || routes < 0) {
        throw std::invalid_argument("staging_overlay: slots 0..1024, layer 0..127, routes >= 0");
    }
    staging_overlay_kernel<<<1, kThreads, 0, stream>>>(table, tags, slots, first_slot, epoch, layer,
                                                       ids, routes, out, stats);
    CUDA_CHECK(cudaGetLastError());
}

void write_staging_word(cudaStream_t stream, std::uint32_t* word, std::uint32_t value) {
    const CUresult r = cuStreamWriteValue32(reinterpret_cast<CUstream>(stream),
                                            reinterpret_cast<CUdeviceptr>(word), value,
                                            CU_STREAM_WRITE_VALUE_DEFAULT);
    if (r != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(r, &name);
        throw std::runtime_error(std::string("write_staging_word: cuStreamWriteValue32 failed: ") +
                                 (name != nullptr ? name : "?"));
    }
}

} // namespace ninfer::expert_cache
