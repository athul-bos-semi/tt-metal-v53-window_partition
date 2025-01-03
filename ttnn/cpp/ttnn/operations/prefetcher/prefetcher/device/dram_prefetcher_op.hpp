// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "tt_metal/include/tt_metal/global_circular_buffer.hpp"

namespace ttnn::operations::dram_prefetcher {

operation::ProgramWithCallbacks dram_prefetcher_multi_core(
    const std::vector<Tensor>& tensors,
    const Tensor& tensor_addrs,
    const uint32_t num_layers,
    const std::shared_ptr<const tt::tt_metal::v1::experimental::GlobalCircularBuffer>& global_cb);

struct DramPrefetcher {
    const Tensor tensor_addrs = Tensor();
    const std::shared_ptr<const tt::tt_metal::v1::experimental::GlobalCircularBuffer> global_cb;
    const uint32_t num_layers;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple("tensor_addrs", "num_layers");
    const auto attribute_values() const { return std::forward_as_tuple(this->tensor_addrs, this->num_layers); }
};

}  // namespace ttnn::operations::dram_prefetcher
