// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "cpp/ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_softmax_backward {

struct MorehSoftmaxBackward {
    static Tensor invoke(
        const Tensor& output_tensor,
        const Tensor& output_grad_tensor,
        uint32_t dim,
        const std::optional<Tensor>& input_grad_tensor,
        const MorehSoftmaxBackwardOp op,
        const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
struct MorehSoftminBackward {
    static Tensor invoke(
        const Tensor& output_tensor,
        const Tensor& output_grad_tensor,
        uint32_t dim,
        const std::optional<Tensor>& input_grad_tensor,
        const MorehSoftmaxBackwardOp op,
        const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
struct MorehLogSoftmaxBackward {
    static Tensor invoke(
        const Tensor& output_tensor,
        const Tensor& output_grad_tensor,
        uint32_t dim,
        const std::optional<Tensor>& input_grad_tensor,
        const MorehSoftmaxBackwardOp op,
        const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_softmax_backward

namespace ttnn {
constexpr auto moreh_softmax_backward = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_softmax_backward",
    ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackward>();
constexpr auto moreh_softmin_backward = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_softmin_backward",
    ttnn::operations::moreh::moreh_softmax_backward::MorehSoftminBackward>();
constexpr auto moreh_logsoftmax_backward = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_logsoftmax_backward",
    ttnn::operations::moreh::moreh_softmax_backward::MorehLogSoftmaxBackward>();
}  // namespace ttnn
