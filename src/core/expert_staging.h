#pragma once

// Staging slots of the routed-expert cache (T25, docs/capy-flash/development-log.md): device slots outside the
// ExpertCache slot table that receive *predicted* experts during a window, so a layer can use an
// expert whose copy landed after the window began without changing the table the window started
// with (ExpertCache's per-window consistency is untouched).
//
// Protocol (one device):
//   * every window has an epoch (1..kMaxStagingEpoch); begin of window: the compute stream writes
//     it to a device word (write_staging_word) before anything of the window is enqueued;
//   * an expert e copied for layer j into staging slot s: the copy on a dedicated stream, then on
//     that stream write_staging_word(tags[s], staging_tag(epoch, j, e)) - stream order makes the
//     tag visible only after the copy completed;
//   * layer j, right after its router: staging_overlay() writes the layer's effective slot row =
//     the table row, plus slot s for every expert e the table does not hold whose tag matches
//     (current epoch, j, e) at that moment. The CPU path (mailbox publish) and the expert kernel
//     both read that row, so they agree on which experts run where, whatever lands afterwards.
// A tag names the window and the layer: a staged expert is only ever used by the layer and window
// it was copied for. A slot may be rewritten only for a later layer of the same window (after the
// device has left the layer that read it) or in a later window; a late copy's tag never matches
// those, so a slot is never read while it is rewritten. At the epoch wrap the owner clears every
// tag with no copy in flight.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::expert_cache {

inline constexpr std::uint32_t kStagingTagEmpty  = 0xFFFFFFFFU;
inline constexpr std::uint32_t kMaxStagingEpoch  = 0xFFFEU; // 16 bits, kStagingTagEmpty unused
inline constexpr std::int32_t kMaxStagingLayers  = 128;     // 7 bits
inline constexpr std::int32_t kStagingExperts    = 512;     // 9 bits
inline constexpr std::int32_t kMaxStagingSlots   = 1024;    // per overlay call
inline constexpr std::int32_t kStagingStatsWords = 4;

[[nodiscard]] constexpr std::uint32_t staging_tag(std::uint32_t epoch, std::int32_t layer,
                                                  std::int32_t expert) noexcept {
    return (epoch << 16) | (static_cast<std::uint32_t>(layer) << 9) |
           static_cast<std::uint32_t>(expert);
}

/**
 * Effective slot row of `layer` for one call (CUDA Graph capturable; reads the tags and the epoch
 * when it executes):
 *   out[e] = table[e]                                   if table[e] >= 0
 *          = first_slot + s                             if tags[s] == staging_tag(*epoch, layer, e)
 *          = -1                                         otherwise
 * for e < 512 (table, out: I32 [512]; tags: `slots` words). `ids` (I32 [routes], the call's routed
 * ids) only feed `stats` (may be host-mapped, kStagingStatsWords words, overwritten): {epoch,
 * staged experts of this layer that had landed, distinct routed experts served from staging,
 * routes served from staging}.
 */
void staging_overlay(const std::int32_t* table, const std::uint32_t* tags, std::int32_t slots,
                     std::int32_t first_slot, const std::uint32_t* epoch, std::int32_t layer,
                     const std::int32_t* ids, std::int32_t routes, std::int32_t* out,
                     std::uint32_t* stats, cudaStream_t stream);

// Stream-ordered 32-bit write of `value` to device word `word` (a stream memory operation: no SM,
// the value is fixed at the call). Capturable.
void write_staging_word(cudaStream_t stream, std::uint32_t* word, std::uint32_t value);

} // namespace ninfer::expert_cache
