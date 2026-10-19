// Error paths of the Qwen4Exp expert window protocol (execution/qwen4_experts.h,
// run_expert_window, used by every prefill chunk): whatever step of a window fails, the window is
// closed before the exception reaches the caller, so the next execution (another request's decode
// round) opens its own window normally. CPU only: a recording backend stands for the offload one,
// which refuses a second begin_window while a window is open.
#include "models/qwen3_5/execution/qwen4_experts.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace ninfer;
using namespace ninfer::models::qwen3_5::execution;

enum class Step { None, Enqueue, Observe, Wait, End };

class RecordingExperts final : public Qwen4RoutedExperts {
public:
    Step fail = Step::None;
    bool open = false;
    int waits = 0, abandons = 0, ends = 0;

    void run(std::int32_t, const Qwen4ExpertBank&, const Tensor&, const Tensor&, const Tensor&,
             Tensor&, WorkspaceArena&, cudaStream_t) override {}

    [[nodiscard]] bool graph_capturable() const noexcept override { return true; }

    void begin_window(cudaStream_t, Qwen4Window) override {
        if (open) { throw std::logic_error("window already open"); }
        open = true;
    }

    void observe_window(cudaStream_t) override {
        if (!open) { throw std::logic_error("no open window"); }
        if (fail == Step::Observe) { throw std::runtime_error("observe failed"); }
    }

    void end_window(cudaStream_t) override {
        if (!open) { throw std::logic_error("no open window"); }
        open = false; // as Qwen4OffloadExperts: closed first, then a failure is reported
        ++ends;
        if (fail == Step::End) { throw std::runtime_error("end failed"); }
    }

    void abandon_window(cudaStream_t) noexcept override {
        if (open) { ++abandons; }
        open = false;
    }
};

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void window_fails_at(Step step, const char* label) {
    RecordingExperts experts;
    experts.fail = step;
    bool thrown  = false;
    try {
        run_expert_window(
            experts, nullptr, Qwen4Window::Prompt,
            [&] {
                if (step == Step::Enqueue) { throw std::runtime_error("enqueue failed"); }
            },
            [&] {
                ++experts.waits;
                if (step == Step::Wait && experts.waits == 1) {
                    throw std::runtime_error("wait failed");
                }
            });
    } catch (const std::runtime_error&) { thrown = true; }
    check(thrown, std::string(label) + ": the failure reaches the caller");
    check(!experts.open, std::string(label) + ": the window is closed");
    check(experts.waits >= 1, std::string(label) + ": the device is waited for before closing");
    // The next execution (an unrelated request's round) opens its window normally.
    experts.fail = Step::None;
    bool next_ok = true;
    try {
        run_expert_window(experts, nullptr, Qwen4Window::Decode, [] {}, [] {});
    } catch (const std::exception&) { next_ok = false; }
    check(next_ok && !experts.open, std::string(label) + ": the next window runs");
    std::cout << "  " << label << ": thrown " << thrown << ", open after " << experts.open
              << ", waits " << experts.waits << ", abandons " << experts.abandons
              << ", next window ok " << next_ok << '\n';
}

} // namespace

int main() {
    window_fails_at(Step::Enqueue, "enqueue fails");
    window_fails_at(Step::Observe, "observe fails");
    window_fails_at(Step::Wait, "device wait fails");
    window_fails_at(Step::End, "end fails after closing");
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "qwen4 expert window: all checks passed\n";
    return 0;
}
