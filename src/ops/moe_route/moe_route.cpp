// ninfer::ops - moe_route / moe_combine_shared / moe_accumulate_shared: validate the public contract, carve workspace,
// launch.
#include "ninfer/ops/moe_route.h"

#include "ops/moe_route/launch.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be contiguous and non-empty with the expected dtype");
}

void require_routed(const Tensor& t, DType dtype, std::int32_t columns, const char* op,
                    const char* label) {
    require_dense(t, dtype, op, label);
    require(t.ne[0] == kMoeRouteTopK && t.ne[1] == columns && t.ne[2] == 1 && t.ne[3] == 1, op,
            std::string(label) + " must be [10,T]");
}

constexpr std::size_t align256(std::size_t value) { return (value + 255) / 256 * 256; }

} // namespace

std::size_t moe_route_workspace_bytes(std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("moe_route_workspace_bytes: T <= 0"); }
    return align256(static_cast<std::size_t>(tokens) * (kMoeRouteExperts + 1) * sizeof(float)) +
           256;
}

void moe_route(const Tensor& x, const Tensor& router, WorkspaceArena& workspace, Tensor& ids,
               Tensor& weights, Tensor& shared_gate, cudaStream_t stream,
               const Tensor* valid_columns, MoeRouteColumns routed_columns) {
    constexpr const char* op = "moe_route";
    require_dense(x, DType::BF16, op, "x");
    require(x.ne[0] == kMoeRouteHidden && x.ne[2] == 1 && x.ne[3] == 1, op, "x must be [2560,T]");
    require_dense(router, DType::BF16, op, "router");
    require(router.numel() == static_cast<std::int64_t>(kMoeRouteExperts + 1) * kMoeRouteHidden,
            op, "router must be BF16 [2560,513]");
    require_routed(ids, DType::I32, x.ne[1], op, "ids");
    require_routed(weights, DType::FP32, x.ne[1], op, "weights");
    require_dense(shared_gate, DType::FP32, op, "shared gate");
    require(shared_gate.numel() == x.ne[1], op, "shared gate must be [T]");
    require(reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 &&
                reinterpret_cast<std::uintptr_t>(router.data) % 16 == 0,
            op, "x and router must be 16-byte aligned");
    if (valid_columns != nullptr) {
        require_dense(*valid_columns, DType::I32, op, "valid columns");
        require(valid_columns->ne[1] == 1 && valid_columns->ne[2] == 1 &&
                    valid_columns->ne[3] == 1 && x.ne[1] % valid_columns->ne[0] == 0,
                op, "valid columns must be [B] with B dividing T");
    }
    require(valid_columns != nullptr || routed_columns == MoeRouteColumns::ValidPrefix, op,
            "LastValid routing needs valid columns");
    auto scope   = workspace.scope();
    auto* logits = static_cast<float*>(workspace
                                           .alloc_bytes(static_cast<std::size_t>(x.ne[1]) *
                                                        (kMoeRouteExperts + 1) * sizeof(float))
                                           .data);
    detail::moe_route_launch(x, router, logits, ids, weights, shared_gate, valid_columns,
                             routed_columns == MoeRouteColumns::LastValid, stream);
}

void moe_combine_shared(const Tensor& partial, const Tensor& shared_gate, const Tensor& shared,
                        Tensor& y, cudaStream_t stream) {
    constexpr const char* op = "moe_combine_shared";
    require_dense(partial, DType::FP32, op, "partial");
    require_dense(shared, DType::BF16, op, "shared");
    require_dense(y, DType::BF16, op, "y");
    require_dense(shared_gate, DType::FP32, op, "shared gate");
    const std::int32_t rows = partial.ne[0], columns = partial.ne[1];
    require(partial.ne[2] == 1 && partial.ne[3] == 1 && shared.ne[0] == rows &&
                shared.ne[1] == columns && shared.numel() == partial.numel() &&
                y.ne[0] == rows && y.ne[1] == columns && y.numel() == partial.numel() &&
                shared_gate.numel() == columns,
            op, "partial/shared/y must be [H,T] and shared_gate [T]");
    detail::moe_combine_shared_launch(partial, shared_gate, shared, y, stream);
}

void moe_accumulate_shared(const Tensor& shared_gate, const Tensor& shared, Tensor& partial,
                           cudaStream_t stream) {
    constexpr const char* op = "moe_accumulate_shared";
    require_dense(partial, DType::FP32, op, "partial");
    require_dense(shared, DType::BF16, op, "shared");
    require_dense(shared_gate, DType::FP32, op, "shared gate");
    const std::int32_t rows = partial.ne[0], columns = partial.ne[1];
    require(partial.ne[2] == 1 && partial.ne[3] == 1 && shared.ne[0] == rows &&
                shared.ne[1] == columns && shared.numel() == partial.numel() &&
                shared_gate.numel() == columns,
            op, "partial/shared must be [H,T] and shared_gate [T]");
    detail::moe_accumulate_shared_launch(shared_gate, shared, partial, stream);
}

} // namespace ninfer::ops
