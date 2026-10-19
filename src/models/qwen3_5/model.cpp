#include "models/qwen3_5/model.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights, DeviceWeights bound,
             int device_count, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), device_count_(device_count),
      resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

std::size_t Model::checked_device(int device) const {
    if (device < 0 || device >= device_count_) {
        throw std::out_of_range("model device " + std::to_string(device) + " is outside the " +
                                std::to_string(device_count_) + "-device backing");
    }
    return static_cast<std::size_t>(device);
}

bool Model::resident(WeightId id, int device) const {
    return !bound_[checked_device(device)].at(id.index).view.parts.empty();
}

const BoundWeight& Model::weight(WeightId id, int device) const {
    const auto& bound = bound_[checked_device(device)].at(id.index);
    if (bound.view.parts.empty()) {
        throw std::invalid_argument(bound.name + " is not resident on device " +
                                    std::to_string(device));
    }
    return bound;
}

const BoundWeight& Model::file_weight(WeightId id) const {
    const auto& bound = bound_[0].at(id.index);
    if (bound.mapped.empty()) {
        throw std::invalid_argument(bound.name + " is not a file-mapped weight");
    }
    return bound;
}

std::uint64_t RoutedExpertSpans::bytes() const noexcept {
    std::uint64_t total = 0;
    for (const auto* bank : {&gate_up, &down}) {
        for (const auto& span : bank->spans) { total += span.bytes; }
    }
    return total;
}

namespace {

// The parent rows [begin, begin + count) one bound matrix covers, given it is a single region of
// whole parent rows.
ExpertBankSpans bank_rows(const BoundWeight& weight) {
    if (weight.view.parts.size() != 1 || weight.view.shape.size() != 2) {
        throw std::invalid_argument(weight.name + ": a routed expert binds one region of a bank");
    }
    const auto& region = weight.view.parts.front();
    const auto& g      = region.parent->geometry;
    if (g.shape.size() != 2 || g.shape[1] != weight.view.shape[1] || region.begin % g.shape[1] ||
        region.end % g.shape[1]) {
        throw std::invalid_argument(weight.name + ": a routed expert is not whole bank rows");
    }
    return {region.parent, region.begin / g.shape[1], (region.end - region.begin) / g.shape[1], {}};
}

} // namespace

RoutedExpertSpans routed_expert_spans(const Model& model, const MoeWeights& moe,
                                      std::uint32_t expert, int device) {
    const auto& weights = moe.experts.at(expert);
    auto gate           = bank_rows(model.weight(weights.gate, device));
    const auto up       = bank_rows(model.weight(weights.up, device));
    auto down           = bank_rows(model.weight(weights.down, device));
    if (up.parent != gate.parent || up.row_begin != gate.row_begin + gate.row_count) {
        throw std::invalid_argument("routed expert " + std::to_string(expert) +
                                    ": gate and up are not adjacent rows of one bank");
    }
    gate.row_count += up.row_count;
    gate.spans = weight_row_block_spans(gate.parent->geometry, gate.row_begin, gate.row_count);
    down.spans = weight_row_block_spans(down.parent->geometry, down.row_begin, down.row_count);
    return {std::move(gate), std::move(down)};
}

ops::WeightInput Model::input(WeightUseId id, int device) const {
    const auto& parameter = weight(id.parameter, device);
    const auto& use       = parameter.uses.at(id.use_index);
    return {parameter.view, use.policy, use.activation_input_divisor};
}

ops::WeightInput Model::input(WeightId id, int device) const {
    if (weight(id, device).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0}, device);
}

} // namespace ninfer::models::qwen3_5
