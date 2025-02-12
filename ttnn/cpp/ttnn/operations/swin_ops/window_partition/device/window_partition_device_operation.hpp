// SPDX-FileCopyrightText: © 2025 BOS
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::swin_ops {

enum class WindowPartParallelizationStrategy { MULTI_CORE, SHARDED_MULTI_CORE };

struct WindowPartDeviceOperation {
    const uint32_t window_size;
    std::vector<uint32_t> resolution;
    const MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    WindowPartParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

// Ref: https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
Tensor windowpart_impl(
    const std::vector<Tensor> &input_tensors,
    const std::uint32_t window_size,
    std::vector<uint32_t> resolution,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace ttnn::operations::swin_ops
