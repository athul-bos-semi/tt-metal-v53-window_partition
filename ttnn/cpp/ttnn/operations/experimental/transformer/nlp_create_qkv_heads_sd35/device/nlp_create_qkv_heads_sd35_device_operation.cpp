// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_sd35_device_operation.hpp"

#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::transformer {

// Hard-coded for SD35
void NlpCreateHeadsSD35DeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_padded_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Error");

    TT_FATAL(input_shape[2] % tt::constants::TILE_HEIGHT == 0, "Error");
    TT_FATAL(input_shape[3] % tt::constants::TILE_HEIGHT == 0, "Error");
    // TT_FATAL((input_shape == tt::tt_metal::LegacyShape({input_shape[0], 1, input_shape[2], 2304})), "Unsupported
    // input shape");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
}

std::vector<ttnn::TensorSpec> NlpCreateHeadsSD35DeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    if (output_mem_config.is_sharded()) {
        TT_ASSERT(false);
        return {};
    }

    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_padded_shape();
    const auto head_dim = 64;                    // head_dim is hard-coded = 64
    auto num_heads = input_shape[3] / head_dim;  // head_dim is hard-coded = 64
    TensorSpec spec(
        Shape({input_shape[0], num_heads, input_shape[2], head_dim}),
        TensorLayout(input_tensor.get_dtype(), PageConfig(Layout::TILE), output_mem_config));
    return {spec, spec, spec};
}

operation::ProgramWithCallbacks NlpCreateHeadsSD35DeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    return multi_core_nlp_create_qkv_heads_sd35(input_tensor, output_tensors, compute_with_storage_grid_size);
}
}  // namespace ttnn::operations::experimental::transformer
