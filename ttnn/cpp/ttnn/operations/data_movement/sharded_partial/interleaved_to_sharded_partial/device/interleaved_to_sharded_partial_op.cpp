// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_partial_op.hpp"

#include "tt_metal/host_api.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/sharded/sharded_op.hpp"


namespace ttnn::operations::data_movement {

void InterleavedToShardedPartialDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    // Validate output tensor
    TT_FATAL(slice_index >= 0 && slice_index < num_slices, "Slice index and num_slices don't match! Index = {} num_slices = {}", slice_index, num_slices);
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Currently, only tile layout is supported for partial I->S");
    TT_FATAL((input_tensor.volume() / input_tensor.get_legacy_shape()[-1]) % num_slices == 0, "Total height of a tensor must be divisible by num_slices!");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");


    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Input tensor must be Interleaved");
    if (input_tensor.get_dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.get_layout() == Layout::TILE);
    }
    auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(this->grid_size.x <= device_grid.x && this->grid_size.y <= device_grid.y, "Grid size for sharding must be less than or equal to total grid available");
    // Divisibility of num_cores and shard size with tensor shape is done in tensor creation, so no need to assert here
}


std::vector<tt::tt_metal::Shape> InterleavedToShardedPartialDeviceOperation::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    tt::tt_metal::Shape shape = input_tensor.get_legacy_shape();

    uint32_t total_height = input_tensor.volume() / shape[-1];
    uint32_t new_height = total_height / this->num_slices;

    shape[0] = 1;
    shape[1] = 1;
    shape[2] = new_height;
    return {shape};
}


std::vector<Tensor> InterleavedToShardedPartialDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto mem_config = this->output_mem_config;
    mem_config.shard_spec = this->shard_spec;
    return {create_device_tensor(
        this->compute_output_shapes(input_tensors).at(0),
        this->output_dtype,
        input_tensor.get_layout(),
        input_tensor.device(),
        mem_config
        )};
}

operation::ProgramWithCallbacks InterleavedToShardedPartialDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    //Will move with sharded ops
    return tt::tt_metal::interleaved_to_sharded_multi_core(input_tensor, output_tensor, this->num_slices, this->slice_index);
}


}
