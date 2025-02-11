// SPDX-FileCopyrightText: © 2025 BOS
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "windowpart.hpp"

namespace ttnn::operations::swin_ops::detail {
namespace py = pybind11;

void bind_windowpart(py::module& module) {
    const auto doc = R"doc(
        What's Up, Doc?
        *proceeds to munch Carrots*
    )doc";

    using OperationType = decltype(ttnn::swin_window_partition);
    ttnn::bind_registered_operation(
        module,
        ttnn::swin_window_partition,
        doc,
        ttnn::pybind_overload_t {
            [] (const OperationType& self,
                const std::vector<ttnn::Tensor>& input_tensors,
                const uint32_t window_size,
                std::vector<uint32_t>& resolution,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                const std::optional<ttnn::Tensor> &optional_output_tensor,
                uint8_t queue_id) {
                    return self(queue_id, input_tensors, window_size, resolution, memory_config, optional_output_tensor);
                },
                py::arg("input_tensors"),
                py::arg("window_size"),
                py::arg("resolution"),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor").noconvert() = std::nullopt,
                py::arg("queue_id") = 0,
        });
}


}  // namespace ttnn::operations::data_movement::detail
