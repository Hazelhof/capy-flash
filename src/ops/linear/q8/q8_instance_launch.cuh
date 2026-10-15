#pragma once
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

#include "ops/linear/q8/q8_output.cuh"

namespace ninfer::ops::detail {
template <class Geometry, int Capacity, class Schedule>
void launch_q8_a16_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Instance = typename Schedule::template with_problem<Geometry::kInputRows, Capacity>;
    q8_with_output(out, Geometry::kOutputRows, [&](auto output) {
        launch_q8_a16_sliced_k_mma<Instance>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}
} // namespace ninfer::ops::detail
