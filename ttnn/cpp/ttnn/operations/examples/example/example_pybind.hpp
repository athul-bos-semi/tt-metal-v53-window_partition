// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/examples/example/example.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::examples {

void bind_example_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::example,
        R"doc(example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

        // Add pybind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::example)& self, const ttnn::Tensor& input_tensor, const uint8_t& queue_id)
                -> ttnn::Tensor { return self(queue_id, input_tensor); },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::examples
