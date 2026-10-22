#include "models/qwen3_5/execution/reasoning_eos_guard.h"

#include "core/device.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

ReasoningEosGuardConfig& guard_config() {
    static ReasoningEosGuardConfig config;
    return config;
}

// One counter in mapped pinned host memory: every device adds to it in place, the host reads it
// without synchronizing (a replacement is visible once its kernel has completed).
unsigned long long* replacement_counter() {
    static unsigned long long* counter = [] {
        void* host = nullptr;
        CUDA_CHECK(cudaHostAlloc(&host, sizeof(unsigned long long),
                                 cudaHostAllocMapped | cudaHostAllocPortable));
        *static_cast<unsigned long long*>(host) = 0;
        return static_cast<unsigned long long*>(host);
    }();
    return counter;
}

} // namespace

void configure_reasoning_eos_guard(ReasoningEosGuardConfig config) {
    if (config.enabled && (config.think_open.empty() || (config.banned0 < 0 && config.banned1 < 0))) {
        throw std::invalid_argument("reasoning EOS guard needs the thinking opener and a terminal id");
    }
    if (config.enabled) {
        (void)replacement_counter();
        std::fprintf(stderr, "reasoning EOS guard: first token of a thinking reply, terminal ids %d %d, opener",
                     config.banned0, config.banned1);
        for (const TokenId token : config.think_open) { std::fprintf(stderr, " %d", token); }
        std::fprintf(stderr, "\n");
    }
    guard_config() = std::move(config);
}

bool reasoning_eos_guard_applies(std::span<const int> prompt) noexcept {
    const ReasoningEosGuardConfig& config = guard_config();
    if (!config.enabled || prompt.size() < config.think_open.size()) { return false; }
    return std::equal(config.think_open.begin(), config.think_open.end(),
                      prompt.end() - static_cast<std::ptrdiff_t>(config.think_open.size()));
}

void guard_first_token(Tensor& logits, Tensor& token, std::int32_t token_domain,
                       const ops::SamplingConfig* configs, const Tensor& logical_positions,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    const ReasoningEosGuardConfig& config = guard_config();
    if (!config.enabled || configs == nullptr) { return; }
    ops::replace_banned_token(logits, token, token_domain, configs, logical_positions,
                              config.banned0, config.banned1, replacement_counter(), workspace,
                              stream);
}

std::uint64_t reasoning_eos_guard_replacements() noexcept {
    if (!guard_config().enabled) { return 0; }
    return *static_cast<volatile unsigned long long*>(replacement_counter());
}

} // namespace ninfer::models::qwen3_5::execution
