import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_test(device, torch_input_tensor_a, torch_input_tensor_b, shard_type, dims, shard_x, shard_y, input_dtype):
    if input_dtype == ttnn.bfloat16:
        size = 2
    if input_dtype == ttnn.bfloat4_b:
        size = 0.5
    if input_dtype == ttnn.bfloat8_b:
        size = 1
    if dims[0] * dims[1] / shard_x / shard_y * size >= 400000:
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

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=input_dtype,
    )

    output = ttnn.add(input_tensor_a, input_tensor_b, memory_config=sharded_mem_config)


@pytest.mark.parametrize(
    "dims", [(2 * 32, 2 * 32), (8 * 32, 32), (4 * 32, 16 * 32), (64 * 32, 32), (32 * 32, 64 * 32), (32 * 32, 128 * 32)]
)
# @pytest.mark.parametrize("dims", [(4 * 32, 16 * 32), (16 * 32, 4 * 32)])
@pytest.mark.parametrize("mem_config", ["height", "width", "block"])
# @pytest.mark.parametrize("mem_config", ["block"])
def test_add_with_block_sharding(device, dims, mem_config):
    torch.manual_seed(0)
    h = dims[0]
    w = dims[1]
    input_dtype = ttnn.bfloat16

    iterations = 1
    for _ in range(iterations):  # Loop over the number of iterations
        h = dims[0]
        w = dims[1]
        torch_input_tensor_a = torch.rand((h, w))
        torch_input_tensor_b = torch.rand((h, w))

        if mem_config == "height":
            if h >= 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 1, 1, input_dtype)
            if h >= 2 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 1, 2, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 2, 1, input_dtype)
            if h >= 4 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 1, 4, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 2, 2, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 4, 1, input_dtype)
            if h >= 8 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 1, 8, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 2, 4, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 4, 2, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 8, 1, input_dtype)
            if h >= 16 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 2, 8, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 4, 4, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 8, 2, input_dtype)
            if h >= 32 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 4, 8, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 8, 4, input_dtype)
            if h >= 64 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "height", dims, 8, 8, input_dtype)
        if mem_config == "width":
            if w >= 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 1, 1, input_dtype)
            if w >= 2 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 1, 2, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 2, 1, input_dtype)
            if w >= 4 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 1, 4, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 2, 2, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 4, 1, input_dtype)
            if w >= 8 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 1, 8, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 2, 4, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 4, 2, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 8, 1, input_dtype)
            if w >= 16 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 2, 8, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 4, 4, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 8, 2, input_dtype)
            if w >= 32 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 4, 8, input_dtype)
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 8, 4, input_dtype)
            if w >= 64 * 32:
                run_test(device, torch_input_tensor_a, torch_input_tensor_b, "width", dims, 8, 8, input_dtype)
        if mem_config == "block":
            if h >= 32:
                if w >= 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 1, 1, input_dtype)
                if w >= 2 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 2, 1, input_dtype)
                if w >= 4 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 4, 1, input_dtype)
                if w >= 8 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 8, 1, input_dtype)
            if h >= 2 * 32:
                if w >= 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 1, 2, input_dtype)
                if w >= 2 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 2, 2, input_dtype)
                if w >= 4 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 4, 2, input_dtype)
                if w >= 8 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 8, 2, input_dtype)
            if h >= 4 * 32:
                if w >= 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 1, 4, input_dtype)
                if w >= 2 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 2, 4, input_dtype)
                if w >= 4 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 4, 4, input_dtype)
                if w >= 8 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 8, 4, input_dtype)
            if h >= 8 * 32:
                if w >= 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 1, 8, input_dtype)
                if w >= 2 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 2, 8, input_dtype)
                if w >= 4 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 4, 8, input_dtype)
                if w >= 8 * 32:
                    run_test(device, torch_input_tensor_a, torch_input_tensor_b, "block", dims, 8, 8, input_dtype)
