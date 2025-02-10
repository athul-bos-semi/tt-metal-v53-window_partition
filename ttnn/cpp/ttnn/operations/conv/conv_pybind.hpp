// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include "conv2d/conv2d_pybind.hpp"
#include "conv_transpose2d/conv_transpose2d_pybind.hpp"
#include "conv3d/conv3d_pybind.hpp"
namespace ttnn::operations::conv {

void py_module(pybind11::module& module) {
    ttnn::operations::conv::conv2d::py_bind_conv2d(module);
    ttnn::operations::conv::conv_transpose2d::py_bind_conv_transpose2d(module);
    ttnn::operations::conv::conv3d::py_bind_conv3d(module);
}
}  // namespace ttnn::operations::conv
