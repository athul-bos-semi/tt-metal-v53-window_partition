# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_sizes": [(1,)],
    "height": [384, 1024],
    "width": [1024, 4096],
    "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "weight": [-0.5, 0, 0.01, 0.5],
}


def skip(
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    weight,
) -> Tuple[bool, Optional[str]]:
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Skipped as ROW_MAJOR_LAYOUT not supported"
    return False, None


def torch_prelu(x, *args, **kwargs):
    weight = kwargs.pop("scalar")
    result = F.prelu(x, torch.tensor(weight, dtype=x.dtype))
    return result


def run(
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    weight,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)

    low = -100.0
    high = 100.0

    torch_input_tensor = torch_random(input_shape, low, high, dtype=torch.float32)
    torch_output_tensor = torch_prelu(torch_input_tensor, scalar=weight)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )

    output_tensor = ttnn.prelu(input_tensor, weight, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
