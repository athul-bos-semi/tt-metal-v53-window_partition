import pytest
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from functools import partial
from models.utility_functions import torch_random
import argparse
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "dims",
    [
        (32, 32),  # 1
        (10 * 32, 32),  # 10
        (10 * 32, 10 * 32),  # 100
        (100 * 32, 10 * 32),  # 1k
        (100 * 32, 20 * 32),  # 2k
        (100 * 32, 50 * 32),  # 5k
        (100 * 32, 80 * 32),  # 8k
        (100 * 32, 100 * 32),  # 10k
        (32 * 500, 32 * 25),  # 12.5k
    ],
)
@pytest.mark.parametrize("input_mem_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("out_mem_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b])
# @pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT])
def test_relu_interleaved(device, dims, input_mem_config, out_mem_config, input_dtype):
    for i in range(10):
        h = dims[0]
        w = dims[1]
        torch_input_tensor_a = torch.rand((h, w))

        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            memory_config=input_mem_config,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=input_dtype,
        )

        output = ttnn.relu(input_tensor_a, memory_config=out_mem_config)
