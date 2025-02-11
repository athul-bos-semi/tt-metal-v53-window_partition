// SPDX-FileCopyrightText: © 2025 BOS
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "tt_metal/common/math.hpp"

#include "ttnn/cpp/ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"

#include "ttnn/cpp/ttnn/operations/swin_ops/window_partition/device/window_partition_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/swin_ops/window_partition/windowpart.hpp"

#include <ranges>

// Toggle this to enable debugging
constexpr bool debug_windowpart = true;
inline void windowpart_db_print(bool condition, const std::string& msg) {
    if constexpr (debug_windowpart) {
        if (condition) {
            std::cout << "[DEBUG] Window Partition: " << msg << std::endl;
        }
    }
}

namespace ttnn::operations::swin_ops {

ttnn::Tensor WindowPartOperation::invoke(
    uint8_t queue_id,
    const std::vector<ttnn::Tensor>& input_tensors,
    const uint32_t window_size,
    std::vector<uint32_t> resolution,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor> optional_output_tensor) {
    
    TT_FATAL(!optional_output_tensor.has_value(), "optional output tensor currently unsupported!");
    const auto mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG); 

    // const auto &input_tensor = ttnn::to_memory_config(input_tensors.at(0), mem_config, std::nullopt);
    // input_tensors[0] = input_tensor;

    return windowpart_impl(input_tensors, window_size, resolution, mem_config);
}

ttnn::Tensor WindowPartOperation::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const uint32_t window_size,
    std::vector<uint32_t> resolution,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor> optional_output_tensor) {
    return invoke(DefaultQueueId, input_tensors, window_size, resolution, memory_config, optional_output_tensor);
}

}  // namespace ttnn::operation::swin_ops
