
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/binary_ng/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations {
namespace unary {
struct UnaryWithParam;
}

namespace binary_ng {

template <BinaryOpType binary_op_type>
struct BinaryNg {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor_a,
        float scalar,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor_a,
        float scalar,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);
};

}  // namespace binary_ng

}  // namespace ttnn::operations
namespace ttnn::experimental {
constexpr auto add = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::add",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::ADD>>();
}
