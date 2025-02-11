// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/decorators.hpp"

#include <ranges>


namespace ttnn {
namespace operations {
namespace swin_ops {

struct WindowPartOperation {

    // Wrapper for TTDNN
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const std::vector<ttnn::Tensor>& input_tensors,
        const uint32_t window_size,
        std::vector<uint32_t> resolution,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        const uint32_t window_size,
        std::vector<uint32_t> resolution,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace swin_ops
}  // namespace operations

constexpr auto swin_window_partition =
    ttnn::register_operation_with_auto_launch_op<"ttnn::swin_window_partition", ttnn::operations::swin_ops::WindowPartOperation>();

}  // namespace ttnn
