#pragma once

// Pinned / pageable split of the HostMapped parents (v0.7 pinned fallback).
//
// The HostMapped parents (Qwen4Exp routed-expert banks, `--embedding-host`) normally share one
// mapped page-locked allocation. Where the page-locked memory a process may hold is capped below
// their size (Windows WDDM, also under WSL2: about half of the physical RAM), the parents that may
// live in pageable memory (HostMappedPlacement::pageable_rank > 0) are moved there, the lowest
// ranks staying pinned first. Readers of a pageable parent: the CPU reads it in place; device
// copies go through pinned staging; no kernel reads it in place.
//
// Pure planning, no CUDA: the materializer allocates what this returns.

#include "artifact/materializer.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::artifact {

struct HostResidency {
    // Per placement, in plan order: 1 = pageable allocation, 0 = pinned allocation.
    std::vector<std::uint8_t> pageable;
    // Per placement: offset in its own allocation (alignment kept, plan order kept).
    std::vector<std::uint64_t> offset;
    std::uint64_t pinned_bytes   = 0; // size of the pinned allocation (0: none)
    std::uint64_t pageable_bytes = 0; // size of the pageable allocation (0: none)
    // The pinned bytes of the parents that must stay pinned (pageable_rank 0) alone.
    std::uint64_t required_pinned_bytes = 0;
    std::size_t pageable_objects        = 0;

    [[nodiscard]] bool split() const noexcept { return pageable_objects != 0; }
};

// Splits `placements` (a plan's host_mapped_objects, `total_bytes` = its host_mapped_bytes) so
// that the pinned allocation holds at most `pinned_budget` bytes where possible: every
// pageable_rank 0 parent stays pinned; then, rank by rank ascending, all parents of one rank are
// pinned together when they still fit (a rank that does not fit is skipped, a later one may fit).
// pinned_budget >= total_bytes returns the plan's own layout: nothing pageable, the plan's offsets,
// pinned_bytes == total_bytes. When the rank 0 parents alone exceed the budget the result keeps
// them pinned anyway (required_pinned_bytes > pinned_budget): the caller refuses.
[[nodiscard]] HostResidency plan_host_residency(std::span<const HostMappedPlacement> placements,
                                                std::uint64_t total_bytes,
                                                std::uint64_t pinned_budget);

// True when some placement may be pageable (pageable_rank > 0).
[[nodiscard]] bool any_pageable_candidate(std::span<const HostMappedPlacement> placements) noexcept;

} // namespace ninfer::artifact
