// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include "tt_metal/common/constants.hpp"

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::dram_prefetcher {

void DramPrefetcher::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() > 0, "Must have at least one input tensor");
    TT_FATAL(this->num_layers > 0, "Prefetcher must run for at least 1 layer");
    TT_FATAL(global_cb.has_value(), "Global circular buffer must be provided");

    uint32_t num_receiver_cores = global_cb->receiver_cores().num_cores();
    for (const auto& tensor : input_tensors) {
        // Check that all tensors are on the same device
        TT_FATAL(tensor.device() == input_tensors[0].device(), "All tensors must be on the same device");
        TT_FATAL(tensor.get_layout() == Layout::TILE, "All tensors must be tilized");
        TT_FATAL(
            tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensors must be width sharded");
        TT_FATAL(tensor.memory_config().buffer_type == BufferType::DRAM, "Input tensors must be in DRAM");

        // Check that all tensors' k is divisible by number of cores in global CB receiver
        TT_FATAL(
            tensor.get_legacy_shape()[1] % num_receiver_cores == 0,
            "All tensors' k must be divisible by the number of receiver cores = {}.",
            num_receiver_cores);

        tt::DataFormat tensor_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.get_dtype());
        TT_FATAL(
            tensor_data_format == tt::DataFormat::Bfp4_b || tensor_data_format == tt::DataFormat::Bfp8_b ||
                tensor_data_format == tt::DataFormat::Float16_b,
            "Input tensors must be of type Bfp4_b, Bfp8_b, or Float16_b");
    }

    // Check that global_cb sender_receiver_core_mapping has same number of receivers for each sender core
    auto sender_receiver_core_mapping = global_cb->sender_receiver_core_mapping();
    for (const auto& [sender_core, receiver_core_range] : sender_receiver_core_mapping) {
        TT_FATAL(
            receiver_core_range.size() == sender_receiver_core_mapping.begin()->second.size(),
            "Global circular buffer must have same number of receivers for each sender core");
    }

    TT_FATAL(
        this->tensor_addrs.device() == input_tensors[0].device(),
        "tensors_addrs must be on the same device as the input tensors");
    TT_FATAL(this->tensor_addrs.get_layout() == Layout::ROW_MAJOR, "Tensor containing addresses must be row major");
    TT_FATAL(
        this->tensor_addrs.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Tensor containing addresses must be height sharded");
    TT_FATAL(
        this->tensor_addrs.memory_config().buffer_type == BufferType::L1, "Tensor containing addresses must be in L1");

    tt::DataFormat tensor_addrs_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_addrs.get_dtype());
    TT_FATAL(tensor_addrs_data_format == tt::DataFormat::UInt32, "Tensor containing addresses must be of type UInt32");
}
// TODO: Remove output tensor entirely (if possible)
std::vector<ttnn::SimpleShape> DramPrefetcher::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {ttnn::SimpleShape{32, 32}};
}
std::vector<Tensor> DramPrefetcher::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    auto output_tensor = create_device_tensor(
        ttnn::SimpleShape{32, 32},
        input_tensors[0].dtype(),
        input_tensors[0].layout(),
        input_tensors[0].device(),
        MemoryConfig{});
    std::vector<Tensor> output_tensors = {output_tensor};
    return output_tensors;
}
operation::ProgramWithCallbacks DramPrefetcher::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return dram_prefetcher_multi_core(input_tensors, this->tensor_addrs, this->num_layers, this->global_cb.value());
}

}  // namespace ttnn::operations::dram_prefetcher
