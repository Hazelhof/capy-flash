// ninfer::ops - PLE wrappers: validate the public contract, then launch.
#include "ninfer/ops/ple.h"

#include "ops/ple/launch.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be contiguous non-empty " +
                (dtype == DType::I32    ? "I32"
                 : dtype == DType::FP32 ? "FP32"
                                        : "BF16"));
}

void require_lanes(const Tensor& slots, std::int32_t lanes, const char* op, const char* label) {
    require_dense(slots, DType::I32, op, label);
    require(slots.numel() == lanes, op, std::string(label) + " must hold one slot per lane");
}

} // namespace

void ple_gate(const Tensor& residual, const Tensor& key_value, const Tensor& query_norm,
              const Tensor& key_norm, std::int32_t streams, float eps, Tensor& gated,
              cudaStream_t stream) {
    constexpr const char* op = "ple_gate";
    require(streams > 0 && streams <= 16, op, "streams must be in [1,16]");
    require_dense(key_value, DType::BF16, op, "key_value");
    require(key_value.ne[2] == 1 && key_value.ne[3] == 1 && key_value.ne[0] % (streams + 1) == 0,
            op, "key_value must be [(S+1)*H,T]");
    const std::int64_t wide = static_cast<std::int64_t>(key_value.ne[0]) / (streams + 1) * streams;
    for (const Tensor* t : {&residual, static_cast<const Tensor*>(&gated)}) {
        require_dense(*t, DType::BF16, op, "wide operand");
        require(t->ne[0] == wide && t->ne[1] == key_value.ne[1] && t->ne[2] == 1 && t->ne[3] == 1,
                op, "residual and gated must be [S*H,T]");
    }
    for (const Tensor* t : {&query_norm, &key_norm}) {
        require_dense(*t, DType::BF16, op, "norm weight");
        require(t->numel() == wide, op, "norm weights must be [S*H]");
    }
    require(eps > 0.0F && std::isfinite(eps), op, "eps must be positive and finite");
    for (const Tensor* t : {&residual, &key_value, &query_norm, &key_norm}) {
        require(t->data != gated.data, op, "gated must not overlap an input");
    }
    detail::ple_gate_launch(residual, key_value, query_norm, key_norm, streams, eps, gated, stream);
}

void ple_dilated_conv(const Tensor& normed, const Tensor& gated, const Tensor& weight,
                      std::int32_t dilation, const Tensor& states, const Tensor& source_slots,
                      Tensor& residual, cudaStream_t stream) {
    constexpr const char* op = "ple_dilated_conv";
    for (const Tensor* t : {&normed, &gated, static_cast<const Tensor*>(&residual)}) {
        require_dense(*t, DType::BF16, op, "column operand");
        require(t->ne[0] == normed.ne[0] && t->ne[1] == normed.ne[1] && t->ne[2] == normed.ne[2],
                op, "column operands must be [C,W,B]");
    }
    require_dense(weight, DType::BF16, op, "weight");
    require(weight.ne[0] == normed.ne[0] && weight.ne[1] >= 2 && weight.ne[2] == 1, op,
            "weight must be [K,C] stored as ne={C,K}");
    const std::int32_t history = (weight.ne[1] - 1) * dilation;
    require(dilation > 0 && history <= 32, op, "history must be in [1,32] columns");
    require_dense(states, DType::BF16, op, "states");
    require(states.ne[0] == history * normed.ne[0], op, "states must be [history*C, S]");
    require_lanes(source_slots, normed.ne[2], op, "source_slots");
    detail::ple_dilated_conv_launch(normed, gated, weight, dilation, states, source_slots, residual,
                                    stream);
}

void ple_conv_advance(const Tensor& normed, const Tensor* valid_columns, std::int32_t history,
                      Tensor& states, const Tensor& source_slots, const Tensor& destination_slots,
                      cudaStream_t stream) {
    constexpr const char* op = "ple_conv_advance";
    require_dense(normed, DType::BF16, op, "normed");
    require_dense(states, DType::BF16, op, "states");
    require(history > 0 && history <= 32 && states.ne[0] == history * normed.ne[0], op,
            "states must be [history*C, S]");
    require_lanes(source_slots, normed.ne[2], op, "source_slots");
    require_lanes(destination_slots, normed.ne[2], op, "destination_slots");
    if (valid_columns != nullptr) { require_lanes(*valid_columns, normed.ne[2], op, "valid"); }
    detail::ple_conv_advance_launch(normed, valid_columns, history, states, source_slots,
                                    destination_slots, stream);
}

} // namespace ninfer::ops
