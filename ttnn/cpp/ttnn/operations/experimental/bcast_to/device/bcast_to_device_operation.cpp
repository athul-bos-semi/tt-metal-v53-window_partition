// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_to_device_operation.hpp"

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::broadcast_to {
BcastToOperation::program_factory_t BcastToOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    switch (input.get_layout()) {
        case Layout::TILE: return BcastToTileFactory{};
        default: TT_THROW("BcastTo: Unsupported input layout");
    }
}

void validate(
    const BcastToOperation::operation_attributes_t& operation_attributes,
    const BcastToOperation::tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    TT_FATAL(input.get_layout() == Layout::TILE, "bcast_to: Input tensor layout must be TILE");
    TT_FATAL(tensor_args.input.storage_type() == StorageType::DEVICE, "bcast_to: Input tensor need to be on device");
    TT_FATAL(tensor_args.input.buffer() != nullptr, "bcast_to: Input tensor need to be allocated in buffers on device");
    if (output.has_value()) {
        TT_FATAL(
            output->get_shape().logical_shape() == operation_attributes.output_shape,
            "bcast_to: Output shape must match operation attributes");
        TT_FATAL(input.get_layout() == output->get_layout(), "bcast_to: Input and output must have same layout");
        TT_FATAL(input.get_dtype() == output->get_dtype(), "bcast_to: Input and output must have same dtype");
        TT_FATAL(input.device() == output->device(), "bcast_to: Input and output must be on the same device");
    }
}

void BcastToOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
};

void BcastToOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
};

BcastToOperation::spec_return_value_t BcastToOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->get_tensor_spec();
    }
    return TensorSpec(
        SimpleShape{operation_attributes.output_shape},
        TensorLayout(
            tensor_args.input.get_dtype(),
            PageConfig(tensor_args.input.get_layout()),
            operation_attributes.memory_config));
};

BcastToOperation::tensor_return_value_t BcastToOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Let's just require it to be allocated ahead of time for now
    if (tensor_args.output.has_value()) {
        return {tensor_args.output.value()};
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

std::tuple<BcastToOperation::operation_attributes_t, BcastToOperation::tensor_args_t> BcastToOperation::invoke(
    const Tensor& input,
    const SmallVector<uint32_t>& output_shape,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config) {
    return {{output_shape, memory_config.value_or(input.memory_config())}, {input, output}};
}
}  // namespace ttnn::operations::experimental::broadcast_to
