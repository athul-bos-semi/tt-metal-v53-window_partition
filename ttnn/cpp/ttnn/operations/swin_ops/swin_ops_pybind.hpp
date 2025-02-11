// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/swin_ops/concat/concat_pybind.hpp"
#include "ttnn/operations/swin_ops/window_partition/windowpart_pybind.hpp"

namespace py = pybind11;

namespace ttnn::operations::swin_ops {

void py_module(py::module& module) {
    detail::bind_concat(module);
    detail::bind_windowpart(module);
}

}
