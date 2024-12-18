# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)

from tests.tt_eager.python_api_testing.unit_testing.misc.test_matmul_1d_gather_in0 import (
    run_multi_core_matmul_1d,
    PREFETCHER_NOC1_GRID,
)

"""
Things to test:
- BFP8
- Different dataformats/shapes
    - Need to add support for multiple output tenosrs
    - Base it off of the input tensor shapes
- Multiple layers
    - Need to change how output tensor is tested?
- Non-square shapes


Testing for writer side:
- Create and output_memory_config (maybe a new arg) across the receiver cores
- Alternative: Replace current output_tensor with output tensor
 sharded on the receiver cores (instead of the sender cores)
  - Requires a new CB (on just the receiver cores), and a new kernel that copies
  data on the global cb (local to the receiver cores) to the output cb on those cores
  -

"""


def num_cores_to_rectangle_grid(num_cores, device):
    """
    Find a rectangular core grid size, given an number of cores.

    Return None if rectangle grid is not possible.
    """
    x = device.compute_with_storage_grid_size().x
    while x > 0 and num_cores % x != 0:
        x -= 1

    if x == 0:
        return None

    y = num_cores // x
    return (x, y)


def get_core_ranges(num_reader_cores, num_global_cb_receivers):
    all_dram_cores = [
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(1, 0),
        ttnn.CoreCoord(2, 0),
        ttnn.CoreCoord(3, 0),
        ttnn.CoreCoord(4, 0),
        ttnn.CoreCoord(5, 0),
        ttnn.CoreCoord(6, 0),
        ttnn.CoreCoord(7, 0),
        ttnn.CoreCoord(8, 0),
        ttnn.CoreCoord(9, 0),
        ttnn.CoreCoord(10, 0),
        ttnn.CoreCoord(11, 0),
    ]
    all_sender_cores = [
        ttnn.CoreCoord(0, 9),
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(0, 4),
        ttnn.CoreCoord(0, 5),
        ttnn.CoreCoord(4, 0),
        ttnn.CoreCoord(4, 9),
        ttnn.CoreCoord(4, 1),
        ttnn.CoreCoord(4, 7),
        ttnn.CoreCoord(4, 6),
        ttnn.CoreCoord(4, 2),
        ttnn.CoreCoord(4, 4),
        ttnn.CoreCoord(4, 5),
    ]
    if num_global_cb_receivers == 2:
        all_receiver_cores_list = [
            (1, 9),
            (2, 9),
            (1, 0),
            (2, 0),
            (1, 4),
            (2, 4),
            (1, 5),
            (2, 5),
            (5, 0),
            (6, 0),
            (5, 9),
            (6, 9),
            (5, 1),
            (6, 1),
            (5, 7),
            (6, 7),
            (5, 6),
            (6, 6),
            (5, 2),
            (6, 2),
            (5, 4),
            (6, 4),
            (5, 5),
            (6, 5),
        ]
    else:
        all_receiver_cores_list = [
            (1, 9),
            # (2, 9),
            (1, 0),
            # (2, 0),
            (1, 4),
            # (2, 4),
            (1, 5),
            # (2, 5),
            (5, 0),
            # (6, 0),
            (5, 9),
            # (6, 9),
            (5, 1),
            # (6, 1),
            (5, 7),
            # (6, 7),
            (5, 6),
            # (6, 6),
            (5, 2),
            # (6, 2),
            (5, 4),
            # (6, 4),
            (5, 5),
            # (6, 5),
        ]
    all_receiver_cores = [
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 9),
                    ttnn.CoreCoord(2, 9) if num_global_cb_receivers == 2 else ttnn.CoreCoord(1, 9),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 0),
                    ttnn.CoreCoord(2, 0) if num_global_cb_receivers == 2 else ttnn.CoreCoord(1, 0),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 4),
                    ttnn.CoreCoord(2, 4) if num_global_cb_receivers == 2 else ttnn.CoreCoord(1, 4),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 5),
                    ttnn.CoreCoord(2, 5) if num_global_cb_receivers == 2 else ttnn.CoreCoord(1, 5),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 0),
                    ttnn.CoreCoord(6, 0) if num_global_cb_receivers == 2 else ttnn.CoreCoord(5, 0),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 9),
                    ttnn.CoreCoord(6, 9) if num_global_cb_receivers == 2 else ttnn.CoreCoord(5, 9),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 1),
                    ttnn.CoreCoord(6, 1) if num_global_cb_receivers == 2 else ttnn.CoreCoord(5, 1),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 7),
                    ttnn.CoreCoord(6, 7) if num_global_cb_receivers == 2 else ttnn.CoreCoord(5, 7),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 6),
                    ttnn.CoreCoord(6, 6) if num_global_cb_receivers == 2 else ttnn.CoreCoord(5, 6),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 2),
                    ttnn.CoreCoord(6, 2) if num_global_cb_receivers == 2 else ttnn.CoreCoord(5, 2),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 4),
                    ttnn.CoreCoord(6, 4) if num_global_cb_receivers == 2 else ttnn.CoreCoord(5, 4),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 5),
                    ttnn.CoreCoord(6, 5) if num_global_cb_receivers == 2 else ttnn.CoreCoord(5, 5),
                ),
            ]
        ),
    ]

    worker_cores_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    mm_optimised_ring_cores = [
        (6, 9),
        (6, 7),
        (6, 6),
        (6, 5),
        (6, 4),
        (6, 2),
        (6, 1),
        (6, 0),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (2, 9),
        (2, 5),
        (2, 4),
        (2, 0),
        (1, 0),
        (1, 4),
        (1, 5),
        (1, 9),
    ]

    mm_optimised_ring_cores = PREFETCHER_NOC1_GRID[::-1]
    hop_grid = [
        (3, 6),
    ]

    dram_cores = all_dram_cores[:num_reader_cores]
    sender_cores = all_sender_cores[:num_reader_cores]
    receiver_cores_list = all_receiver_cores_list[: num_reader_cores * num_global_cb_receivers]
    # receiver_cores_list = all_receiver_cores_list[:num_reader_cores]
    receiver_cores = all_receiver_cores[:num_reader_cores]

    return (
        dram_cores,
        sender_cores,
        receiver_cores_list,
        receiver_cores,
        worker_cores_range_set,
        mm_optimised_ring_cores,
        hop_grid,
    )


@pytest.mark.parametrize(
    "num_reader_cores, num_tensors, input_shapes, dtypes, num_layers",
    [
        (2, 3, [(128, 128), (128, 128 * 2), (128, 128 * 3)], [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16], 2),
        (2, 2, [(256, 512), (256, 512)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(1024, 256), (1024, 256)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(128, 128), (128, 128)], [ttnn.bfloat4_b] * 2, 2),
        (2, 2, [(256, 1024), (256, 1024)], [ttnn.bfloat4_b] * 2, 5),
        (
            12,
            5,
            [(2304, 3840)] * 5,
            [ttnn.bfloat4_b] * 5,
            2,
        ),  # FF1/3 = 72 tiles x 120 tiles = 8640 tiles / 24 cores = 720 tiles per receiver core
        (
            1,
            4,
            [(192, 320), (192, 320), (192, 320), (192, 320)],
            [ttnn.bfloat4_b, ttnn.bfloat8_b] * 2,
            1,
        ),
        (12, 5, [(3840, 2304)] * 5, [ttnn.bfloat8_b] * 5, 5),  # FF2
        (12, 6, [(2304, 1536)] * 6, [ttnn.bfloat8_b] * 6, 5),  # QKV
        (12, 5, [(2304, 2304)] * 5, [ttnn.bfloat8_b] * 5, 5),  # DO
        # Takes really long to set up
        (
            12,
            5,
            [(2304, 3840), (3840, 2304), (2304, 3840), (2304, 1536), (2304, 2304)],
            [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat8_b],
            80,
        ),  # ff1 + ff2 +ff3+ qkv + do
    ],
)
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_run_prefetcher(
    device,
    num_tensors,
    input_shapes,
    num_layers,
    num_reader_cores,
    dtypes,
    use_program_cache,
    function_level_defaults,
):
    logger.info(f"Running test_run_prefetcher with num_tensors={num_tensors}, input_shape={input_shapes[0]}")
    assert len(input_shapes) == len(dtypes)
    assert num_tensors == len(input_shapes)

    num_global_cb_receivers = 2

    K, N = input_shapes[0]

    (
        dram_cores,
        sender_cores,
        receiver_cores_list,
        receiver_cores,
        worker_cores_range_set,
        mm_optimised_ring_cores,
        hop_grid,
    ) = get_core_ranges(num_reader_cores, num_global_cb_receivers)

    if num_reader_cores != 12:
        mm_optimised_ring_cores = receiver_cores_list

    receiver_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in receiver_cores_list
        ]
    )

    print(f"sender_cores: {sender_cores}")
    print(f"receiver_cores: {receiver_cores}")
    print(f"receiver_cores_list: {receiver_cores_list}")

    sender_receiver_mapping = list(zip(sender_cores, receiver_cores))

    # FF1 is 368640 per receiver core = 360 tiles
    global_cb_size = 512 * 512 * 4
    # global_cb_size = 360 * 576 * 4 # 4*FF1 in bfp4
    global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, global_cb_size)
    print(f"global cb size {global_cb_size}")

    ##### Set up the input tensors #####
    dram_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord) for core_coord in dram_cores])
    sender_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord) for core_coord in sender_cores])

    pt_tensors = []
    for l in range(num_layers):
        for t in range(num_tensors):
            pt_tensors.append(torch.randn(input_shapes[t]))

    tt_tensors_all = []

    for tid in range(num_tensors * num_layers):
        K, N = input_shapes[tid % num_tensors]
        input_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_core_range_set,
                [K, N // len(dram_cores)],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        tt_tensor = ttnn.as_tensor(
            pt_tensors[tid],
            device=device,
            dtype=dtypes[tid % num_tensors],
            memory_config=input_sharded_mem_config,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_tensors_all.append(tt_tensor)
    tt_tensors = tt_tensors_all[:num_tensors]

    # Set up the tensor addrs
    # TODO: Fix when greater than a tile size
    tensor_addrs = torch.tensor([x.buffer_address() for x in tt_tensors_all])
    tensor_addrs = tensor_addrs.repeat(len(dram_cores), 1)
    tensor_addrs_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            sender_core_range_set,
            [tensor_addrs.shape[0] // len(dram_cores), tensor_addrs.shape[1]],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    tt_tensor_addrs = ttnn.as_tensor(
        tensor_addrs, device=device, dtype=ttnn.uint32, memory_config=tensor_addrs_mem_config
    )

    ##### Output mem config #####
    reader_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            sender_core_range_set,
            [
                32,  # K * num_tensors,
                32,  # N // len(sender_cores),
            ],  # Assuming all tensors have the same shape TODO: extend to different shapes
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    writer_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            receiver_core_range_set,
            [K * num_tensors, N // receiver_core_range_set.num_cores()],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    ##### Setup up sub devices #####
    prefetcher_sub_device = ttnn.SubDevice([sender_core_range_set])
    worker_sub_device = ttnn.SubDevice([worker_cores_range_set])
    sub_device_manager = device.create_sub_device_manager([prefetcher_sub_device, worker_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)
    worker_sub_device_id = 1  # Can we parameterize this?

    max_dst_tiles = 8
    grid = receiver_cores_list
    num_cores = grid[0] * grid[1] if isinstance(grid, tuple) else len(grid)
    storage_grid = num_cores_to_rectangle_grid(num_cores, device)
    M = 32

    in0_shapes = []
    out_shapes = []
    block_dims = []
    for tid in range(num_tensors):
        K, N = input_shapes[tid]
        in0_shape = [1, 1, M, K]
        in0_shapes.append(in0_shape)
        out_shape = [1, 1, M, N]
        out_shapes.append(out_shape)

        in0_block_h = M // ttnn.TILE_SIZE
        in0_block_w = K // num_cores // ttnn.TILE_SIZE
        out_block_h = M // ttnn.TILE_SIZE
        out_block_w = N // num_cores // ttnn.TILE_SIZE

        out_subblock_h = 1
        out_subblock_w = max_dst_tiles if (out_block_h == 1 and out_block_w <= max_dst_tiles) else 4
        while out_block_w % out_subblock_w != 0:
            out_subblock_w -= 1

        logger.debug("in0 block h w " + str(in0_block_h) + " " + str(in0_block_w))
        logger.debug("in1 block h w " + str(in0_block_w) + " " + str(out_block_w))
        logger.debug("out block h w " + str(out_block_h) + " " + str(out_block_w))
        logger.debug("out subblock h w " + str(out_subblock_h) + " " + str(out_subblock_w))

        block_dim = [in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w]
        block_dims.append(block_dim)
    # x, y
    if isinstance(grid, tuple):  # Generate random grid
        CORE_RANGE = [(x, y) for y in range(storage_grid[1]) for x in range(storage_grid[0])]
        random.shuffle(CORE_RANGE)
    else:
        CORE_RANGE = grid

    input_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in mm_optimised_ring_cores
        ]
    )

    output_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in CORE_RANGE
        ]
    )

    hop_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in hop_grid
        }
    )

    print(f"num_cores: {num_cores}")

    output_mem_configs = []
    for shape in out_shapes:
        _, _, M, N = shape

        output_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                output_core_range_set,
                [M, N // num_cores],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        output_mem_configs.append(output_sharded_mem_config)

    in0_tensors = []
    in0_t_tensors = []
    for shape in in0_shapes:
        in0 = torch.randn(shape)
        in0_tensors.append(in0)

        _, _, M, K = shape

        in0_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                input_core_range_set,
                [M, K // num_cores],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        in0_t = ttnn.from_torch(
            in0,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=in0_sharded_mem_config,
            sub_device_ids=[ttnn.SubDeviceId(worker_sub_device_id)],
        )
        in0_t_tensors.append(in0_t)

    program_configs = []
    for block_dim in block_dims:
        in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w = block_dim
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=storage_grid,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            num_global_cb_receivers=num_global_cb_receivers,
            hop_cores=hop_core_range_set,  # Only use with noc1 grid
        )
        program_configs.append(program_config)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    ttnn.dram_prefetcher(
        tt_tensors,
        tt_tensor_addrs,
        num_layers,
        global_circular_buffer,
        reader_output_mem_config,
        writer_output_mem_config,
    )
    all_passing = True

    # FIXME: RESULTS IN HANG FOR LARGE SHAPES
    for l in range(num_layers):
        outputs_t = []
        for t in range(num_tensors):
            idx = l * num_tensors + t
            logger.info(f"Running matmul for layer {l}, tensor {t}")

            output_t = ttnn.matmul(
                in0_t_tensors[t],
                tt_tensors_all[idx],
                program_config=program_configs[t],
                memory_config=output_mem_configs[t],
                compute_kernel_config=compute_kernel_config,
                global_cb=global_circular_buffer,
            )
            outputs_t.append(output_t)

        for t in range(num_tensors):
            idx = l * num_tensors + t
            logger.info(f"Checking matmul for layer {l}, tensor {t}")
            tt_out = ttnn.to_torch(outputs_t[t], sub_device_ids=[ttnn.SubDeviceId(worker_sub_device_id)])
            pt_out = in0_tensors[t] @ pt_tensors[idx]

            dtype = dtypes[t]
            if dtype == ttnn.bfloat4_b:
                pcc_threshold = 0.99
            elif dtype == ttnn.bfloat8_b:
                pcc_threshold = 0.999
            elif dtype == ttnn.bfloat16:
                pcc_threshold = 0.999

            passing, output = comp_pcc(pt_out, tt_out, pcc_threshold)
            logger.info(output)
            all_passing = passing and all_passing

    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)

    assert all_passing
