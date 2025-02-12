// SPDX-FileCopyrightText: © 2025 BOS
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/common/logger.hpp"

#include "ttnn/cpp/ttnn/operations/swin_ops/window_partition/device/window_partition_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/swin_ops/window_partition/device/multi_core_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::swin_ops {

WindowPartParallelizationStrategy WindowPartDeviceOperation::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    if (input_tensors[0].is_sharded()) {
        return WindowPartParallelizationStrategy::SHARDED_MULTI_CORE;
    } else {
        return WindowPartParallelizationStrategy::MULTI_CORE;
    }
}

void WindowPartDeviceOperation::validate(const std::vector<Tensor> &input_tensors) const {
    tt::log_debug(tt::LogTest, ">>> Validate Entry");
    const auto &input_tensor = input_tensors[0];
    // Number of tensors in input_tensors, which for us would only be 1, can be obtained as input_tensors.size()
    // Shape of the tensor can be obtained as input_tensor.get_legacy_shape()
    TT_FATAL(this->resolution[0] % this->window_size == 0, "Window Partition operation is currently only designed for cases where image height is divisible by window size");
    TT_FATAL(this->resolution[1] % this->window_size == 0, "Window Partition operation is currently only designed for cases where image width is divisible by window size");
    bool sharded_input = input_tensor.is_sharded();
    bool tile_layout_warning = false;
    // Grid of the sharded input can be obtained as input_tensor.shard_spec().value().grid
    // Memory layout of the sharded input can be obtained as input_tensor.memory_config().memory_layout

    TT_FATAL(input_tensor.buffer(), "Operand needs to be allocated in a buffer on device.");
    TT_FATAL(input_tensor.device(), "Operand needs to be on device.");
    if(input_tensor.get_layout() == Layout::TILE) {
        tile_layout_warning = true;
    }
    if (sharded_input) {
        TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR,
            "Only row major supported for sharded tensors currently.");
        TT_FATAL(input_tensor.shard_spec().has_value(), "Sharded tensors must have a shard spec.");
        TT_FATAL(
            input_tensor.memory_config().memory_layout != TensorMemoryLayout::BLOCK_SHARDED,
            "Block sharded inputs are not supported");
    }

    if (sharded_input) {
        const auto memory_layout = input_tensor.memory_config().memory_layout;
        TT_FATAL(
            this->output_mem_config.memory_layout == memory_layout,
            "Sharded output and input must have the same memory layout.");
        TT_FATAL(this->output_mem_config.is_sharded(), "Output must be sharded if input is sharded.");
        TT_FATAL(
            this->output_mem_config.shard_spec.value().grid == input_tensor.shard_spec().value().grid,
            "Sharded output and input must be in the same grid.");
    }
}

std::vector<ttnn::SimpleShape> WindowPartDeviceOperation::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    tt::log_debug(tt::LogTest, ">>> Compute Output Shapes Entry");
    ttnn::SimpleShape shape_out = input_tensors[0].get_logical_shape();
    
    return {shape_out};
}

std::vector<Tensor> WindowPartDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const Tensor &ref_in_tensor = input_tensors.at(0);

    if (this->output_mem_config.is_sharded()) {
        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            ref_in_tensor.get_dtype(),
            ref_in_tensor.get_layout(),
            ref_in_tensor.device(),
            ref_in_tensor.memory_config())};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, ref_in_tensor.get_dtype(), ref_in_tensor.get_layout(), ref_in_tensor.memory_config());
    }
}

operation::ProgramWithCallbacks WindowPartDeviceOperation::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    switch (this->get_parallelization_strategy(input_tensors)) {
        case WindowPartParallelizationStrategy::SHARDED_MULTI_CORE: {
            tt::log_debug(tt::LogTest, ">>> Sharded Multi-core Entry");
            return detail::windowpart_multi_core(input_tensors, this->window_size, this->resolution, output_tensors[0]);
        }
        case WindowPartParallelizationStrategy::MULTI_CORE:
        default: {
            tt::log_debug(tt::LogTest, ">>> Multi-core Entry");
            return detail::windowpart_multi_core(input_tensors, this->window_size, this->resolution, output_tensors[0]);
        }
    };
}

Tensor windowpart_impl(const std::vector<Tensor> &input_tensors, const std::uint32_t window_size, std::vector<uint32_t> resolution, const MemoryConfig &output_mem_config) {
    tt::log_debug(tt::LogTest, ">>> windowpart_impl Entry");
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensors[0]}))};
    operation::launch_op(
        [window_size, resolution, output_mem_config](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) -> std::vector<Tensor> {
                if (input_tensors[0].is_sharded()) {
                    return operation::run(WindowPartDeviceOperation{window_size, resolution, output_mem_config}, {input_tensors});
                } else {
                    // row major should default to row major and tilized to tilized implementations, but the below loop turned RM to tilized when possible
                    Layout target_layout = input_tensors[0].get_layout();
                    std::vector<ttnn::operations::experimental::auto_format::FormatParams> input_format_params;
                    input_format_params.reserve(input_tensors.size());
                    const auto &input_tensor = input_tensors[0];
                    if (target_layout == Layout::ROW_MAJOR) {
                        input_format_params.push_back(ttnn::operations::experimental::auto_format::FormatParams{
                            .pad_shape = input_tensor.get_legacy_shape(),
                            .pad_value = 0.0,
                            .target_layout = target_layout});
                    } else {
                        tt::tt_metal::LegacyShape pad_shape = ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
                        input_format_params.push_back(
                            ttnn::operations::experimental::auto_format::FormatParams{.pad_shape = pad_shape, .pad_value = 0.0, .target_layout = target_layout});
                    }

                    return operation::run_with_autoformat(
                        WindowPartDeviceOperation{window_size, resolution, output_mem_config}, {input_tensors}, {input_format_params}, {target_layout});
                }
            },
        input_tensors,
        output_tensors);
    return output_tensors.at(0);
}

}
