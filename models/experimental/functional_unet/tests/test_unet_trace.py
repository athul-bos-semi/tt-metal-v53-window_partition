# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import ttnn
import pytest

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn

from models.utility_functions import skip_for_grayskull, divup


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 68864, "trace_region_size": 423936}], indirect=True)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((2, 1, 10),),
)
def test_unet_trace(
    batch: int,
    groups: int,
    iterations: int,
    device,
    use_program_cache,
    reset_seeds,
):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=True)

    model = unet_shallow_torch.UNet.from_random_weights(groups=1)
    torch_output_tensor = model(torch_input)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    input_tensor = ttnn.allocate_tensor_on_device(
        ttnn_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    logger.info(f"Compiling model with warmup run")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=0)
    output_tensor = ttnn_model(input_tensor).cpu()

    logger.info(f"Capturing trace")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=0)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = ttnn_model(input_tensor)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    logger.info(f"Running trace for {iterations} iterations...")

    outputs = []
    ttnn.DumpDeviceProfiler(device)
    ttnn.synchronize_device(device)
    start = time.time()
    for _ in range(iterations):
        # ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(output_tensor.cpu(blocking=False))
    ttnn.synchronize_device(device)
    end = time.time()
    logger.info(f"PERF={iterations * batch / (end-start) : .2f} fps")

    logger.info(f"Running sanity check against reference model output")
    B, C, H, W = torch_output_tensor.shape
    ttnn_tensor = ttnn.to_torch(outputs[-1]).reshape(B, H, W, -1)[:, :, :, :C].permute(0, 3, 1, 2)
    assert_with_pcc(torch_output_tensor, ttnn_tensor, 0.986)


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 68864, "trace_region_size": 423936, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((2, 1, 10),),
)
def test_unet_trace_2cq(
    batch: int,
    groups: int,
    iterations: int,
    device,
    use_program_cache,
    reset_seeds,
):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=True)

    model = unet_shallow_torch.UNet.from_random_weights(groups=1)
    torch_output_tensor = model(torch_input)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    model_event = ttnn.create_event(device)
    read_event = ttnn.create_event(device)

    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(ttnn_input.volume() // ttnn_input.shape[-1], dram_grid_size.x),
            ttnn_input.shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    input_tensor = ttnn.allocate_tensor_on_device(
        ttnn_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, dram_memory_config
    )
    ttnn.record_event(0, op_event)
    ttnn.record_event(1, read_event)

    logger.info(f"Compiling model with warmup run")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)

    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)

    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    ttnn.record_event(0, op_event)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(output_tensor.volume() // output_tensor.get_legacy_shape()[-1], dram_grid_size.x),
            output_tensor.get_legacy_shape()[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )
    dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config)
    logger.info(f"Done compile run")

    logger.info(f"Capturing trace")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    ttnn.record_event(0, op_event)

    input_trace_addr = l1_input_tensor.buffer_address()
    shape = l1_input_tensor.shape
    dtype = l1_input_tensor.dtype
    layout = l1_input_tensor.layout
    output_tensor.deallocate(force=True)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)

    # Try allocating our persistent input tensor here and verifying it matches the address that trace captured
    l1_input_tensor = ttnn.allocate_tensor_on_device(
        shape, dtype, layout, device, ttnn_model.input_sharded_memory_config
    )
    assert input_trace_addr == l1_input_tensor.buffer_address()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config, dram_output_tensor)

    outputs = []
    ttnn.DumpDeviceProfiler(device)
    ttnn.synchronize_device(device)
    start = time.time()
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
    ttnn.record_event(1, write_event)
    for _ in range(iterations - 1):
        ttnn.wait_for_event(0, write_event)
        l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config, l1_input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)
        dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config, dram_output_tensor)
        ttnn.record_event(0, model_event)
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(1, model_event)
        outputs.append(dram_output_tensor.cpu(blocking=False, cq_id=1))
        ttnn.record_event(1, read_event)
    ttnn.wait_for_event(0, write_event)
    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config, l1_input_tensor)
    ttnn.record_event(0, op_event)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.wait_for_event(0, read_event)
    dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config, dram_output_tensor)
    ttnn.record_event(0, model_event)
    ttnn.wait_for_event(1, model_event)
    outputs.append(dram_output_tensor.cpu(blocking=False, cq_id=1))
    # ttnn.record_event(1, read_event)
    ttnn.synchronize_device(device)
    end = time.time()

    logger.info(f"PERF={iterations * batch / (end-start) : .2f} fps")

    logger.info(f"Running sanity check against reference model output")
    B, C, H, W = torch_output_tensor.shape
    ttnn_tensor = ttnn.to_torch(outputs[-1]).reshape(B, H, W, -1)[:, :, :, :C].permute(0, 3, 1, 2)
    assert_with_pcc(torch_output_tensor, ttnn_tensor, 0.986)

    ttnn.release_trace(device, tid)
