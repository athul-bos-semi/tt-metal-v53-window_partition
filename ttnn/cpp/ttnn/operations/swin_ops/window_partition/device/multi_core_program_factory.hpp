// SPDX-FileCopyrightText: © 2025 BOS
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/common/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::swin_ops::detail {

// tt::tt_metal::operation::ProgramWithCallbacks sharded_windowpart_multi_core(
//     const std::vector<Tensor> &input_tensors, 
//     const uint32_t window_size, 
//     std::vector<uint32_t> resolution, 
//     Tensor &output);

tt::tt_metal::operation::ProgramWithCallbacks windowpart_multi_core(
    const std::vector<Tensor> &input_tensors, 
    const uint32_t window_size, 
    std::vector<uint32_t> resolution, 
    Tensor &output);

}  // namespace ttnn::operations::swin_ops::detail
