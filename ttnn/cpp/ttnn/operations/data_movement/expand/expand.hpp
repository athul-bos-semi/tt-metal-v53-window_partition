// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ttnn/decorators.hpp"

namespace ttnn::operations::expand {
struct ExpandOperation {
    static Tensor invoke(
        const ttnn::Tensor& input,
        tt::stl::Span<const int32_t> sizes,
        std::optional<MemoryConfig>& memory_config,
        std::optional<uint32_t>& queue_id);
};
}  // namespace ttnn::operations::expand

namespace ttnn {
constexpr auto expand = ttnn::register_operation<"ttnn::expand", ttnn::operations::expand::ExpandOperation>();
}
