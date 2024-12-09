# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def run_sub_devices(device, replicate_sub_devices=False):
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    tensix_cores1 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 4),
            ),
        }
    )
    sub_device_1 = ttnn.SubDevice([tensix_cores0])
    sub_device_2 = ttnn.SubDevice([tensix_cores1])
    sub_devices_1 = [sub_device_1, sub_device_2]
    sub_devices_2 = [sub_device_2]
    if replicate_sub_devices:
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        sub_devices_1 = [sub_devices_1] * num_devices
        sub_devices_2 = [sub_devices_2] * num_devices
    sub_device_manager1 = device.create_sub_device_manager(sub_devices_1, 3200)
    sub_device_manager2 = device.create_sub_device_manager(sub_devices_2, 3200)
    device.load_sub_device_manager(sub_device_manager1)
    ttnn.synchronize_devices(device, sub_device_ids=[ttnn.SubDeviceId(1)])
    ttnn.synchronize_devices(device, sub_device_ids=[ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
    ttnn.synchronize_devices(device)
    device.load_sub_device_manager(sub_device_manager2)
    ttnn.synchronize_devices(device, sub_device_ids=[ttnn.SubDeviceId(0)])
    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager1)
    device.remove_sub_device_manager(sub_device_manager2)


def run_sub_devices_program(device, replicate_sub_devices=False):
    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    if is_mesh_device:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        num_devices = device.get_num_devices()
    else:
        inputs_mesh_mapper = None
        output_mesh_composer = None
        num_devices = 1
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 4),
            ),
        }
    )
    tensix_cores1 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    sub_device_1 = ttnn.SubDevice([tensix_cores0])
    sub_device_2 = ttnn.SubDevice([tensix_cores1])
    sub_devices = [sub_device_1, sub_device_2]
    if replicate_sub_devices:
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        sub_devices = [sub_devices] * num_devices
    sub_device_manager = device.create_sub_device_manager(sub_devices, 3200)
    device.load_sub_device_manager(sub_device_manager)

    x = torch.randn(num_devices, 1, 64, 64, dtype=torch.bfloat16)
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=inputs_mesh_mapper,
        sub_device_ids=[ttnn.SubDeviceId(0)],
    )

    xt_host = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
        sub_device_ids=[ttnn.SubDeviceId(1)],
    )

    ttnn.copy_host_to_device_tensor(xt_host, xt, sub_device_ids=[ttnn.SubDeviceId(1)])

    grid_size = device.compute_with_storage_grid_size()
    shard_size = [32, 64]
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    yt = ttnn.interleaved_to_sharded(
        xt, grid_size, shard_size, shard_scheme, shard_orientation, output_dtype=ttnn.bfloat16
    )
    y = ttnn.to_torch(yt, device=device, mesh_composer=output_mesh_composer, sub_device_ids=[ttnn.SubDeviceId(1)])

    eq = torch.equal(x, y)
    assert eq

    y = ttnn.to_torch(yt.cpu(sub_device_ids=[ttnn.SubDeviceId(0)]), mesh_composer=output_mesh_composer)

    eq = torch.equal(x, y)
    assert eq

    event = ttnn.create_event(device)

    yt2 = ttnn.interleaved_to_sharded(
        xt, grid_size, shard_size, shard_scheme, shard_orientation, output_dtype=ttnn.bfloat16
    )
    ttnn.record_event(0, event, [ttnn.SubDeviceId(1)])
    ttnn.wait_for_event(0, event)
    y2 = ttnn.to_torch(yt2, device=device, mesh_composer=output_mesh_composer, sub_device_ids=[ttnn.SubDeviceId(0)])

    eq = torch.equal(x, y2)
    assert eq

    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_sub_devices(device, enable_async_mode):
    run_sub_devices(device)


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
@pytest.mark.parametrize("replicate_sub_devices", (False, True))
def test_sub_devices_mesh(mesh_device, replicate_sub_devices, enable_async_mode):
    if mesh_device.get_num_devices() == 1:
        pytest.skip("#15833: Skipping test for single device mesh")
    run_sub_devices(mesh_device, replicate_sub_devices)


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_sub_device_program(device, enable_async_mode):
    run_sub_devices_program(device)


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
@pytest.mark.parametrize("replicate_sub_devices", (False, True))
def test_sub_device_program_mesh(mesh_device, replicate_sub_devices, enable_async_mode):
    if mesh_device.get_num_devices() == 1:
        pytest.skip("#15833: Skipping test for single device mesh")
    run_sub_devices_program(mesh_device, replicate_sub_devices)
