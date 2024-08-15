# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from loguru import logger
import ttnn

import pytest

import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
    torch2tt_tensor,
    get_devices_for_t3000,
    tt2torch_tensor,
)
import torch

FF1_HANG_PARAMETRIZATION = (1024, 4608, 18432, 4, 72, 3, 1, 8, 100000)

CHIP_ID_TO_COORDINATES_T3K = [None] * 8
CHIP_ID_TO_COORDINATES_T3K[0] = (1, 0)
CHIP_ID_TO_COORDINATES_T3K[1] = (1, 1)
CHIP_ID_TO_COORDINATES_T3K[2] = (2, 1)
CHIP_ID_TO_COORDINATES_T3K[3] = (2, 0)
CHIP_ID_TO_COORDINATES_T3K[4] = (0, 0)
CHIP_ID_TO_COORDINATES_T3K[5] = (0, 1)
CHIP_ID_TO_COORDINATES_T3K[6] = (3, 1)
CHIP_ID_TO_COORDINATES_T3K[7] = (3, 0)


# Used to reproduce issue #8665 with matmul 2D (Falcon 7b matmuls)
@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (FF1_HANG_PARAMETRIZATION, (1024, 4608, 18432, 4, 72, 3, 1, 1, 100000)),
    ids=["ff1-hang", "ff1-pass"],
)
@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_reproduce_matmul_2d_hang(
    num_devices,
    all_devices,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
    use_program_cache,
    activations_zero_percentage,
    weights_zero_percentage,
    zero_columns,
    zero_rows,
    determinism_check_enabled=False,
    determinism_check_iterations=1,
):
    torch.manual_seed(1234)

    devices = []
    if num_devices == 8:
        devices = get_devices_for_t3000(all_devices, num_devices)
    else:
        devices = all_devices

    print("Running on ", num_devices, " devices")

    in0_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        # Volume must match batch size
                        ttl.tensor.CoreCoord(0, 0),
                        ttl.tensor.CoreCoord(7, 7),
                    ),
                }
            ),
            [
                128,
                576,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    in1_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1)

    in0_dtype = ttl.tensor.DataType.BFLOAT16
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT16

    a_shape = [1, 1, seq_len, inner_dim]
    b_shape = [1, 1, inner_dim, weights_n]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape)

    if zero_columns and weights_zero_percentage > 0:
        raise Exception(
            "You can't set zero percentage of weights and choose core columns to have zero weights; choose one."
        )

    if zero_rows and activations_zero_percentage > 0:
        raise Exception(
            "You can't set zero percentage of activations and choose core rows to have zero weights; choose one."
        )

    if activations_zero_percentage > 0:
        logger.info(f"Activations zero percentage: {activations_zero_percentage}")
        total_elements_a = seq_len * inner_dim
        zero_indices = torch.randperm(total_elements_a)[: int(activations_zero_percentage * total_elements_a / 100)]
        A.view(-1)[zero_indices] = 0

    if weights_zero_percentage > 0:
        logger.info(f"Weights zero percentage: {weights_zero_percentage}")
        total_elements_b = inner_dim * weights_n
        zero_indices = torch.randperm(total_elements_b)[: int(weights_zero_percentage * total_elements_b / 100)]
        B.view(-1)[zero_indices] = 0

    if zero_columns:
        logger.info(f"Zero columns: {zero_columns}")
        for zero_column in zero_columns:
            start = zero_column * per_core_N * 32
            end = start + per_core_N * 32
            B[:, :, :, start:end] = 0

    if zero_rows:
        logger.info(f"Zero rows: {zero_rows}")
        for zero_row in zero_rows:
            start = zero_row * per_core_M * 32
            end = start + per_core_M * 32
            A[:, :, start:end, :] = 0

    a_t = []
    b_t = []

    for device_idx in range(num_devices):
        a_t.append(torch2tt_tensor(A, devices[device_idx], ttl.tensor.Layout.TILE, in0_mem_config, in0_dtype))
        b_t.append(torch2tt_tensor(B, devices[device_idx], ttl.tensor.Layout.TILE, in1_mem_config, in1_dtype))

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )

    compute_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    num_nd_outputs = [0] * num_devices
    out = []
    reference_out = []

    # First run for a reference output
    for device_idx in range(num_devices):
        out.append(
            ttnn.matmul(
                a_t[device_idx],
                b_t[device_idx],
                program_config=program_config,
                memory_config=out_mem_config,
                dtype=out_dtype,
                compute_kernel_config=compute_config,
            )
        )

    if determinism_check_enabled:
        for device_idx in range(num_devices):
            reference_out.append(tt2torch_tensor(out[device_idx]))

    # loop_count iterations to test determinism/hang
    for i in range(loop_count):
        # run matmul on all devices
        for device_idx in range(num_devices):
            out[device_idx].deallocate(True)
            out[device_idx] = ttnn.matmul(
                a_t[device_idx],
                b_t[device_idx],
                program_config=program_config,
                memory_config=out_mem_config,
                dtype=out_dtype,
                compute_kernel_config=compute_config,
            )

        # synchronize devices in the order of all_devices fixture list; this list ensures we first synchronize
        # all local devices and then all remote devices, to avoid misinterpreting a hang on the remote device
        # caused by problems on the local device it is attached to
        for device_idx in range(num_devices):
            if num_devices != 1:
                if num_devices == 2:
                    print("Start sync device id: ", device_idx)
                if num_devices == 8:
                    print(
                        "Start sync device id: ",
                        device_idx,
                        " eth coordinates: ",
                        CHIP_ID_TO_COORDINATES_T3K[device_idx],
                    )
            else:
                print("Start single device sync:")
            ttl.device.Synchronize(all_devices[device_idx])
            if num_devices != 1:
                if num_devices == 2:
                    print("End sync device id: ", device_idx)
                if num_devices == 8:
                    print(
                        "End sync device id: ",
                        device_idx,
                        " eth coordinates: ",
                        CHIP_ID_TO_COORDINATES_T3K[device_idx],
                    )
            else:
                print("End single device sync")

        # check if the output matches the first run output
        if determinism_check_enabled and i % determinism_check_iterations == 0:
            for device_idx in range(num_devices):
                pt_out = tt2torch_tensor(out[device_idx])
                if torch.equal(reference_out[device_idx], pt_out):
                    logger.info(f"Device {device_idx} PCC: 1.0")
                else:
                    # for determinism check, we avoid calling comp_pcc func as it is heavy and with too many operations,
                    # part of the code that replaces nans/infs with zeros starts leaking memory, even if deallocation is forced,
                    # so we call it only in case we see tensors are not equal
                    _, pcc = comp_pcc(reference_out[device_idx], pt_out)
                    logger.info(f"Device {device_idx} PCC: {pcc}")
                    num_nd_outputs[device_idx] += 1

        logger.info(f"Iteration = {i}, done")

    if determinism_check_enabled:
        for device_idx in range(num_devices):
            logger.info(f"Number of non-deterministic outputs on device {device_idx} is {num_nd_outputs[device_idx]}")

    for device_idx in range(num_devices):
        out[device_idx].deallocate(True)

    for device_idx in range(num_devices):
        ttl.device.Synchronize(devices[device_idx])


@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
def test_specific_chip_reproduce_matmul_2d_hang_t3000(
    all_devices,
    logical_chip_index,
    use_program_cache,
    activations_zero_percentage,
    weights_zero_percentage,
    zero_columns,
    zero_rows,
):
    num_devices_t3000 = 8
    if len(all_devices) != num_devices_t3000:
        pytest.skip("Test is only valid for t3000 machines")

    print(
        "Selecting device id: ",
        logical_chip_index,
        " coordinates: ",
        CHIP_ID_TO_COORDINATES_T3K[logical_chip_index],
    )
    target_device = all_devices[logical_chip_index]
    devices = [target_device]

    test_reproduce_matmul_2d_hang(
        1,
        devices,
        1024,
        4608,
        18432,
        4,
        72,
        3,
        1,
        8,
        100000,
        use_program_cache,
        activations_zero_percentage,
        weights_zero_percentage,
        zero_columns,
        zero_rows,
    )


@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (FF1_HANG_PARAMETRIZATION,),
    ids=[
        "ff1-hang",
    ],
)
@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_determinism(
    num_devices,
    all_devices,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
    use_program_cache,
    activations_zero_percentage,
    weights_zero_percentage,
    zero_columns,
    zero_rows,
    determinism_check_iterations,
):
    test_reproduce_matmul_2d_hang(
        num_devices,
        all_devices,
        seq_len,
        inner_dim,
        weights_n,
        per_core_M,
        per_core_N,
        in_block_w,
        out_subblock_h,
        out_subblock_w,
        loop_count,
        use_program_cache,
        activations_zero_percentage,
        weights_zero_percentage,
        zero_columns,
        zero_rows,
        determinism_check_enabled=True,
        determinism_check_iterations=determinism_check_iterations,
    )


@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (FF1_HANG_PARAMETRIZATION,),
    ids=[
        "ff1-hang",
    ],
)
def test_determinism_specific_chip(
    all_devices,
    logical_chip_index,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
    use_program_cache,
    activations_zero_percentage,
    weights_zero_percentage,
    zero_columns,
    zero_rows,
    determinism_check_iterations,
):
    num_devices_t3000 = 8
    if len(all_devices) != num_devices_t3000:
        pytest.skip("Test is only valid for t3000 machines")

    print(
        "Selecting device id: ",
        logical_chip_index,
        " coordinates: ",
        CHIP_ID_TO_COORDINATES_T3K[logical_chip_index],
    )
    target_device = all_devices[logical_chip_index]
    devices = [target_device]

    test_reproduce_matmul_2d_hang(
        1,
        devices,
        seq_len,
        inner_dim,
        weights_n,
        per_core_M,
        per_core_N,
        in_block_w,
        out_subblock_h,
        out_subblock_w,
        loop_count,
        use_program_cache,
        activations_zero_percentage,
        weights_zero_percentage,
        zero_columns,
        zero_rows,
        determinism_check_enabled=True,
        determinism_check_iterations=determinism_check_iterations,
    )
