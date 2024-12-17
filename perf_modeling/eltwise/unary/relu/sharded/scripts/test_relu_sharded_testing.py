import pytest
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from functools import partial
from models.utility_functions import torch_random
import argparse
from tests.ttnn.utils_for_testing import assert_with_pcc


def iterate_per_shard(dims, type):
    h = dims[0]
    w = dims[1]
    if type == "height":
        if h >= 32:
            print("height " + str(h) + "," + str(w))
        if h >= 4 * 32:
            print("height " + str(h // 4) + "," + str(w))
        if h >= 8 * 32:
            print("height " + str(h // 8) + "," + str(w))
        if h >= 32 * 32:
            print("height " + str(h // 32) + "," + str(w))
        if h >= 64 * 32:
            print("height " + str(h // 64) + "," + str(w))
    if type == "width":
        if h >= 32:
            print("width " + str(h) + "," + str(w))
        if h >= 4 * 32:
            print("width " + str(h) + "," + str(w // 4))
        if h >= 8 * 32:
            print("width " + str(h) + "," + str(w // 8))
        if h >= 32 * 32:
            print("width " + str(h) + "," + str(w // 32))
        if h >= 64 * 32:
            print("width " + str(h) + "," + str(w // 64))
    if type == "block":
        if h >= 32:
            if w >= 32:
                print("block " + str(h) + "," + str(w))
            if w >= 4 * 32:
                print("block " + str(h) + "," + str(w // 4))
            if w >= 8 * 32:
                print("block " + str(h) + "," + str(w // 8))
        if h >= 4 * 32:
            if w >= 32:
                print("block " + str(h // 4) + "," + str(w))
            if w >= 4 * 32:
                print("block " + str(h // 4) + "," + str(w // 4))
            if w >= 8 * 32:
                print("block " + str(h // 4) + "," + str(w // 8))
        if h >= 8 * 32:
            if w >= 32:
                print("block " + str(h // 8) + "," + str(w))
            if w >= 4 * 32:
                print("block " + str(h // 8) + "," + str(w // 4))
            if w >= 8 * 32:
                print("block " + str(h // 8) + "," + str(w // 8))


def run_test(device, torch_input_tensor_a, shard_type, dims, shard_x, shard_y, input_dtype):
    if input_dtype == ttnn.bfloat16:
        size = 2
    if input_dtype == ttnn.bfloat4_b:
        size = 0.5
    if input_dtype == ttnn.bfloat8_b:
        size = 1
    if dims[0] * dims[1] / shard_x / shard_y * size >= 600000:
        return
    if shard_type == "height":
        sharded_mem_config = ttnn.create_sharded_memory_config_(
            dims,
            (ttnn.CoreGrid(x=shard_x, y=shard_y)),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
    if shard_type == "width":
        sharded_mem_config = ttnn.create_sharded_memory_config_(
            dims,
            (ttnn.CoreGrid(x=shard_x, y=shard_y)),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
    if shard_type == "block":
        sharded_mem_config = ttnn.create_sharded_memory_config_(
            dims,
            (ttnn.CoreGrid(x=shard_x, y=shard_y)),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=input_dtype,
    )
    output = ttnn.relu(input_tensor_a, memory_config=sharded_mem_config)


@pytest.mark.parametrize("dims", [(32, 32), (4 * 32, 4 * 32), (8 * 32, 8 * 32), (32 * 32, 32 * 32), (64 * 32, 64 * 32)])
@pytest.mark.parametrize("mem_config", ["height", "width", "block"])
@pytest.mark.parametrize("mem_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_relu_sharded(device, dims, mem_config, mem_layout, input_dtype):
    h = dims[0]
    w = dims[1]

    iterations = 10
    for _ in range(iterations):  # Loop over the number of iterations
        torch.manual_seed(0)
        h = dims[0]
        w = dims[1]
        torch_input_tensor_a = torch.rand((h, w))

        if mem_config == "height":
            if h >= 32:
                run_test(device, torch_input_tensor_a, "height", dims, 1, 1, input_dtype)
            if h >= 4 * 32:
                run_test(device, torch_input_tensor_a, "height", dims, 1, 4, input_dtype)
            if h >= 8 * 32:
                run_test(device, torch_input_tensor_a, "height", dims, 1, 8, input_dtype)
            if h >= 32 * 32:
                run_test(device, torch_input_tensor_a, "height", dims, 4, 8, input_dtype)
            if h >= 64 * 32:
                run_test(device, torch_input_tensor_a, "height", dims, 8, 8, input_dtype)
        if mem_config == "width":
            if h >= 32:
                run_test(device, torch_input_tensor_a, "width", dims, 1, 1, input_dtype)
            if h >= 4 * 32:
                run_test(device, torch_input_tensor_a, "width", dims, 1, 4, input_dtype)
            if h >= 8 * 32:
                run_test(device, torch_input_tensor_a, "width", dims, 1, 8, input_dtype)
            if h >= 32 * 32:
                run_test(device, torch_input_tensor_a, "width", dims, 4, 8, input_dtype)
            if h >= 64 * 32:
                run_test(device, torch_input_tensor_a, "width", dims, 8, 8, input_dtype)
        if mem_config == "block":
            if h >= 32:
                if w >= 32:
                    run_test(device, torch_input_tensor_a, "block", dims, 1, 1, input_dtype)
                if w >= 4 * 32:
                    run_test(device, torch_input_tensor_a, "block", dims, 1, 4, input_dtype)
                if w >= 8 * 32:
                    run_test(device, torch_input_tensor_a, "block", dims, 1, 8, input_dtype)
            if h >= 4 * 32:
                if w >= 32:
                    run_test(device, torch_input_tensor_a, "block", dims, 4, 1, input_dtype)
                if w >= 4 * 32:
                    run_test(device, torch_input_tensor_a, "block", dims, 4, 4, input_dtype)
                if w >= 8 * 32:
                    run_test(device, torch_input_tensor_a, "block", dims, 4, 8, input_dtype)
            if h >= 8 * 32:
                if w >= 32:
                    run_test(device, torch_input_tensor_a, "block", dims, 8, 1, input_dtype)
                if w >= 4 * 32:
                    run_test(device, torch_input_tensor_a, "block", dims, 8, 4, input_dtype)
                if w >= 8 * 32:
                    run_test(device, torch_input_tensor_a, "block", dims, 8, 8, input_dtype)
