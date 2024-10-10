# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": [
            [1, 1, 2048],
            [1, 1, 256],
            [1, 1, 3072],
            [1, 1, 4096],
            [1, 1, 768],
            [1, 10, 2048],
            [1, 10, 3072],
            [1, 10, 4096],
            [1, 100, 14, 14],
            [1, 100, 192],
            [1, 1008, 14, 14],
            [1, 1008, 7, 7],
            [1, 1024, 10, 10],
            [1, 1024, 14, 14],
            [1, 1024, 19, 19],
            [1, 1024, 28, 28],
            [1, 1024, 45, 80],
            [1, 1024, 50, 68],
            [1, 1024, 7, 7],
            [1, 104, 28, 28],
            [1, 104, 56, 56],
            [1, 1056, 14, 14],
            [1, 1056, 48, 48],
            [1, 1056, 7, 7],
            [1, 1056, 96, 96],
            [1, 1088, 14, 14],
            [1, 1088, 7, 7],
            [1, 110, 1, 1],
            [1, 1104, 14, 14],
            [1, 1104, 7, 7],
            [1, 112, 1, 1],
            [1, 112, 14, 14],
            [1, 1120, 14, 14],
            [1, 1120, 7, 7],
            [1, 1152, 14, 14],
            [1, 1152, 7, 7],
            [1, 1184, 14, 14],
            [1, 1184, 7, 7],
            [1, 12, 1, 1],
            [1, 120, 1, 1],
            [1, 120, 28, 28],
            [1, 120, 40, 40],
            [1, 120, 56, 56],
            [1, 1200, 14, 14],
            [1, 1200, 7, 7],
            [1, 1216, 14, 14],
            [1, 1216, 7, 7],
            [1, 1232, 14, 14],
            [1, 1232, 28, 28],
            [1, 1248, 14, 14],
            [1, 1248, 7, 7],
            [1, 128, 10, 10],
            [1, 128, 100, 136],
            [1, 128, 112, 112],
            [1, 128, 14, 14],
            [1, 128, 150, 150],
            [1, 128, 17, 17],
            [1, 128, 180, 320],
            [1, 128, 200, 272],
            [1, 128, 28, 28],
            [1, 128, 3, 3],
            [1, 128, 5, 5],
            [1, 128, 56, 56],
            [1, 128, 64, 64],
            [1, 128, 7, 7],
            [1, 128, 75, 75],
            [1, 128, 90, 160],
            [1, 1280, 1, 1],
            [1, 1280, 14, 14],
            [1, 1280, 7, 7],
            [1, 128],
            [1, 1296, 14, 14],
            [1, 1296, 7, 7],
            [1, 12],
            [1, 1312, 14, 14],
            [1, 1312, 7, 7],
            [1, 132, 1, 1],
            [1, 1344, 14, 14],
            [1, 1344, 28, 28],
            [1, 1344, 7, 7],
            [1, 1376, 14, 14],
            [1, 1376, 7, 7],
            [1, 1392, 14, 14],
            [1, 1392, 28, 28],
            [1, 1392, 7, 7],
            [1, 1408, 14, 14],
            [1, 1408, 7, 7],
            [1, 144, 1, 1],
            [1, 144, 14, 14],
            [1, 144, 28, 28],
            [1, 144, 56, 56],
            [1, 144, 7, 7],
            [1, 1440, 14, 14],
            [1, 1440, 7, 7],
            [1, 1472, 14, 14],
            [1, 1472, 7, 7],
            [1, 1488, 14, 14],
            [1, 1488, 7, 7],
            [1, 15, 15, 512],
            [1, 1504, 14, 14],
            [1, 1504, 7, 7],
            [1, 1512, 14, 14],
            [1, 1512, 7, 7],
            [1, 1536, 10, 10],
            [1, 1536, 14, 14],
            [1, 1536, 7, 7],
            [1, 1568, 14, 14],
            [1, 1568, 7, 7],
            [1, 1584, 14, 14],
            [1, 1584, 7, 7],
            [1, 16, 1, 1],
            [1, 16, 112, 112],
            [1, 16, 14, 14],
            [1, 16, 160, 160],
            [1, 16, 224, 224],
            [1, 16, 28, 28],
            [1, 16, 56, 56],
            [1, 160, 14, 14],
            [1, 160, 28, 28],
            [1, 160, 56, 56],
            [1, 160, 7, 7],
            [1, 1600, 14, 14],
            [1, 1600, 7, 7],
            [1, 1632, 14, 14],
            [1, 1632, 7, 7],
            [1, 1664, 14, 14],
            [1, 1664, 7, 7],
            [1, 168, 1, 1],
            [1, 168, 28, 28],
            [1, 168, 56, 56],
            [1, 1680, 14, 14],
            [1, 1680, 7, 7],
            [1, 1696, 14, 14],
            [1, 1696, 7, 7],
            [1, 1728, 14, 14],
            [1, 1728, 7, 7],
            [1, 174, 1, 1],
            [1, 1760, 14, 14],
            [1, 1760, 7, 7],
            [1, 1776, 14, 14],
            [1, 1776, 7, 7],
            [1, 1792, 14, 14],
            [1, 1792, 7, 7],
            [1, 18, 1, 1],
            [1, 18, 14, 14],
            [1, 18, 28, 28],
            [1, 18, 56, 56],
            [1, 1824, 14, 14],
            [1, 1824, 7, 7],
            [1, 1856, 7, 7],
            [1, 1872, 14, 14],
            [1, 1872, 7, 7],
            [1, 1888, 7, 7],
            [1, 192, 14, 14],
            [1, 192, 17, 17],
            [1, 192, 28, 28],
            [1, 192, 35, 35],
            [1, 192, 56, 56],
            [1, 192, 7, 7],
            [1, 192, 8, 8],
            [1, 1920, 14, 14],
            [1, 1920, 7, 7],
            [1, 196, 1, 1],
            [1, 1968, 14, 14],
            [1, 1968, 7, 7],
            [1, 20, 1, 1],
            [1, 2016, 14, 14],
            [1, 2016, 7, 7],
            [1, 2048, 10, 10],
            [1, 2048, 14, 14],
            [1, 2048, 23, 40],
            [1, 2048, 25, 34],
            [1, 2048, 7, 7],
            [1, 2064, 14, 14],
            [1, 2064, 7, 7],
            [1, 208, 14, 14],
            [1, 208, 28, 28],
            [1, 2112, 14, 14],
            [1, 2112, 7, 7],
            [1, 216, 28, 28],
            [1, 216, 56, 56],
            [1, 2160, 7, 7],
            [1, 2208, 7, 7],
            [1, 222, 1, 1],
            [1, 224, 1, 1],
            [1, 224, 112, 112],
            [1, 224, 14, 14],
            [1, 224, 17, 17],
            [1, 224, 28, 28],
            [1, 224, 35, 35],
            [1, 224, 56, 56],
            [1, 224, 7, 7],
            [1, 232, 112, 112],
            [1, 232, 56, 56],
            [1, 24, 1, 1],
            [1, 24, 112, 112],
            [1, 24, 14, 14],
            [1, 240, 1, 1],
            [1, 240, 14, 14],
            [1, 240, 28, 28],
            [1, 240, 56, 56],
            [1, 2520, 14, 14],
            [1, 2520, 7, 7],
            [1, 256, 1, 1],
            [1, 256, 100, 136],
            [1, 256, 112, 112],
            [1, 256, 128, 128],
            [1, 256, 13, 17],
            [1, 256, 14, 14],
            [1, 256, 17, 17],
            [1, 256, 180, 320],
            [1, 256, 19, 19],
            [1, 256, 200, 272],
            [1, 256, 25, 34],
            [1, 256, 28, 28],
            [1, 256, 3, 3],
            [1, 256, 32, 32],
            [1, 256, 38, 38],
            [1, 256, 45, 80],
            [1, 256, 5, 5],
            [1, 256, 50, 68],
            [1, 256, 56, 56],
            [1, 256, 7, 7],
            [1, 256, 7, 9],
            [1, 256, 75, 75],
            [1, 256, 8, 8],
            [1, 256, 90, 160],
            [1, 26, 1, 1],
            [1, 264, 1, 1],
            [1, 288, 14, 14],
            [1, 288, 28, 28],
            [1, 288, 56, 56],
            [1, 2904, 24, 24],
            [1, 2904, 48, 48],
            [1, 30, 1, 1],
            [1, 3024, 14, 14],
            [1, 3024, 7, 7],
            [1, 308, 1, 1],
            [1, 32, 1, 1],
            [1, 32, 112, 112],
            [1, 32, 120, 160],
            [1, 32, 14, 14],
            [1, 32, 147, 147],
            [1, 32, 149, 149],
            [1, 32, 150, 150],
            [1, 32, 192, 192],
            [1, 32, 256, 256],
            [1, 32, 26, 26],
            [1, 32, 28, 28],
            [1, 32, 30, 40],
            [1, 32, 56, 56],
            [1, 32, 60, 80],
            [1, 32, 7, 7],
            [1, 320, 14, 14],
            [1, 320, 17, 17],
            [1, 320, 28, 28],
            [1, 320, 7, 7],
            [1, 320, 8, 8],
            [1, 336, 112, 112],
            [1, 336, 14, 14],
            [1, 336, 28, 28],
            [1, 336, 56, 56],
            [1, 348, 1, 1],
            [1, 352, 14, 14],
            [1, 352, 28, 28],
            [1, 36, 1, 1],
            [1, 36, 14, 14],
            [1, 36, 28, 28],
            [1, 36, 56, 56],
            [1, 3712, 14, 14],
            [1, 3712, 7, 7],
            [1, 384, 14, 14],
            [1, 384, 17, 17],
            [1, 384, 28, 28],
            [1, 384, 56, 56],
            [1, 384, 7, 7],
            [1, 384, 8, 8],
            [1, 4, 14, 14],
            [1, 40, 1, 1],
            [1, 400, 14, 14],
            [1, 400, 7, 7],
            [1, 408, 14, 14],
            [1, 408, 28, 28],
            [1, 4096],
            [1, 416, 14, 14],
            [1, 416, 28, 28],
            [1, 432, 14, 14],
            [1, 432, 28, 28],
            [1, 440, 14, 14],
            [1, 440, 7, 7],
            [1, 448, 14, 14],
            [1, 448, 28, 28],
            [1, 448, 56, 56],
            [1, 448, 8, 8],
            [1, 48, 112, 112],
            [1, 48, 14, 14],
            [1, 48, 56, 56],
            [1, 48, 7, 7],
            [1, 480, 14, 14],
            [1, 480, 28, 28],
            [1, 480, 7, 7],
            [1, 512, 10, 10],
            [1, 512, 100, 136],
            [1, 512, 14, 14],
            [1, 512, 16, 16],
            [1, 512, 19, 19],
            [1, 512, 23, 40],
            [1, 512, 25, 34],
            [1, 512, 28, 28],
            [1, 512, 38, 38],
            [1, 512, 45, 80],
            [1, 512, 50, 68],
            [1, 512, 56, 56],
            [1, 512, 7, 7],
            [1, 512, 8, 8],
            [1, 512, 90, 160],
            [1, 52, 1, 1],
            [1, 528, 14, 14],
            [1, 528, 192, 192],
            [1, 528, 28, 28],
            [1, 528, 96, 96],
            [1, 54, 1, 1],
            [1, 544, 14, 14],
            [1, 544, 7, 7],
            [1, 56, 1, 1],
            [1, 576, 14, 14],
            [1, 576, 28, 28],
            [1, 576, 7, 7],
            [1, 58, 1, 1],
            [1, 60, 28, 28],
            [1, 608, 14, 14],
            [1, 608, 7, 7],
            [1, 624, 14, 14],
            [1, 624, 28, 28],
            [1, 64, 1, 1],
            [1, 64, 112, 112],
            [1, 64, 120, 160],
            [1, 64, 128, 128],
            [1, 64, 14, 14],
            [1, 64, 147, 147],
            [1, 64, 150, 150],
            [1, 64, 160, 160],
            [1, 64, 180, 320],
            [1, 64, 200, 272],
            [1, 64, 224, 224],
            [1, 64, 24, 24],
            [1, 64, 28, 28],
            [1, 64, 30, 40],
            [1, 64, 300, 300],
            [1, 64, 35, 35],
            [1, 64, 360, 640],
            [1, 64, 400, 544],
            [1, 64, 480, 640],
            [1, 64, 56, 56],
            [1, 64, 60, 80],
            [1, 64, 73, 73],
            [1, 64, 80, 80],
            [1, 640, 14, 14],
            [1, 640, 7, 7],
            [1, 64],
            [1, 672, 14, 14],
            [1, 672, 28, 28],
            [1, 672, 56, 56],
            [1, 672, 7, 7],
            [1, 696, 28, 28],
            [1, 696, 56, 56],
            [1, 704, 14, 14],
            [1, 704, 7, 7],
            [1, 72, 1, 1],
            [1, 72, 112, 112],
            [1, 72, 14, 14],
            [1, 72, 28, 28],
            [1, 72, 40, 40],
            [1, 72, 56, 56],
            [1, 72, 80, 80],
            [1, 720, 14, 14],
            [1, 720, 28, 28],
            [1, 726, 1, 1],
            [1, 728, 19, 19],
            [1, 728, 38, 38],
            [1, 736, 14, 14],
            [1, 736, 7, 7],
            [1, 7392, 12, 12],
            [1, 7392, 24, 24],
            [1, 768, 14, 14],
            [1, 768, 28, 28],
            [1, 768, 7, 7],
            [1, 784, 14, 14],
            [1, 784, 7, 7],
            [1, 8, 1, 1],
            [1, 8, 112, 112],
            [1, 80, 1, 1],
            [1, 80, 112, 112],
            [1, 80, 56, 56],
            [1, 800, 14, 14],
            [1, 800, 7, 7],
            [1, 816, 14, 14],
            [1, 832, 14, 14],
            [1, 832, 7, 7],
            [1, 84, 1, 1],
            [1, 864, 14, 14],
            [1, 864, 7, 7],
            [1, 88, 28, 28],
            [1, 888, 14, 14],
            [1, 888, 7, 7],
            [1, 896, 14, 14],
            [1, 896, 28, 28],
            [1, 896, 7, 7],
            [1, 912, 14, 14],
            [1, 912, 7, 7],
            [1, 92, 14, 14],
            [1, 928, 14, 14],
            [1, 928, 7, 7],
            [1, 96, 112, 112],
            [1, 96, 14, 14],
            [1, 96, 28, 28],
            [1, 96, 35, 35],
            [1, 96, 56, 56],
            [1, 96, 71, 71],
            [1, 96, 73, 73],
            [1, 960, 14, 14],
            [1, 960, 7, 7],
            [1, 992, 14, 14],
            [1, 992, 7, 7],
            # [1, "s0", 256],
            # [1, "s0", 768],
            [100, 1, 2048],
            [59, 4096],
            [6, 1, 100, 256],
            [920, 1, 2048],
        ],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    golden_function = ttnn.get_golden_function(ttnn.relu)
    torch_output_tensor = golden_function(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.relu(input_tensor_a, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
