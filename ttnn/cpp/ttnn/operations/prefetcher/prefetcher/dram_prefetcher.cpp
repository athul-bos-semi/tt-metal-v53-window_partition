// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher.hpp"
#include <optional>

#include "device/dram_prefetcher_op.hpp"
#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "tt_metal/include/tt_metal/global_circular_buffer.hpp"

namespace ttnn::operations::dram_prefetcher {

Tensor ExecuteDramPrefetcher::invoke(
    std::vector<ttnn::Tensor>& tensors,
    const Tensor& tensor_addrs,
    const uint32_t num_layers,
    const std::optional<const tt::tt_metal::v1::experimental::GlobalCircularBuffer>& global_cb,
    const MemoryConfig& reader_output_mem_config,
    const MemoryConfig& writer_output_mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output(tensors))};
    operation::launch_op(
        [tensor_addrs, num_layers, global_cb, reader_output_mem_config, writer_output_mem_config](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                DramPrefetcher{tensor_addrs, global_cb, num_layers, reader_output_mem_config, writer_output_mem_config},
                input_tensors);
        },
        tensors,
        output_tensors);

    return output_tensors.at(0);
}

}  // namespace ttnn::operations::dram_prefetcher
