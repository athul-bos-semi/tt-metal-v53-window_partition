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

struct ConcatOperation {

    // Wrapper for TTDNN
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const std::vector<ttnn::Tensor>& input_tensors,
        int dim,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt,
        unsigned int groups = 1);

    static ttnn::Tensor invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        int dim,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt,
        unsigned int groups = 1);
};

}  // namespace swin_ops
}  // namespace operations

constexpr auto bos_concat =
    ttnn::register_operation_with_auto_launch_op<"ttnn::bos_concat", ttnn::operations::swin_ops::ConcatOperation>();

}  // namespace ttnn
