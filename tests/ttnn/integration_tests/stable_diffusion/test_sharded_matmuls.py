# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
import pytest
import ttnn

import tt_lib as ttl
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    determine_largest_subblock_size,
    round_up_to_tile_dim,
)
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor, skip_for_wormhole_b0


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_grayskull()
@pytest.mark.parametrize(
    "W0, Z0, W1, Z1, M, K, N, input_dtype, output_dtype, bias, output_mem_config, input_mem_config, program_config_type, grid_size, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, transpose_mcast, fused_activation, fuse_batch, mcast_in0",
    [
        (
            2,
            1,
            1,
            1,
            32,
            320,
            1280,
            "BFLOAT16",
            "BFLOAT16",
            False,
            "DEV_0_DRAM_INTERLEAVED",
            "DEV_0_L1_INTERLEAVED",
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            (8, 8),
            1,
            2,
            1,
            2,
            1,
            None,
            None,
            True,
            True,
        ),
        (
            2,
            1,
            1,
            1,
            32,
            1280,
            1280,
            "BFLOAT16",
            "BFLOAT16",
            False,
            "DEV_0_DRAM_INTERLEAVED",
            "DEV_0_DRAM_INTERLEAVED",
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            (8, 8),
            1,
            2,
            1,
            2,
            1,
            None,
            None,
            True,
            True,
        ),
        (
            2,
            1,
            1,
            1,
            32,
            1280,
            320,
            "BFLOAT16",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_INTERLEAVED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 8),
            4,
            1,
            2,
            1,
            2,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            8192,
            320,
            320,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 5),
            2,
            2,
            2,
            32,
            2,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            8192,
            320,
            1536,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            2,
            8,
            1,
            32,
            11,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            4096,
            64,
            4096,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_HEIGHT_SHARDED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            (8, 8),
            2,
            1,
            8,
            2,
            128,
            None,
            None,
            True,
            False,
        ),
        (
            1,
            1,
            1,
            1,
            4096,
            4096,
            64,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_HEIGHT_SHARDED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            (8, 8),
            128,
            2,
            2,
            2,
            2,
            None,
            None,
            True,
            False,
        ),
        (
            1,
            1,
            1,
            1,
            8192,
            512,
            320,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_HEIGHT_SHARDED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            (8, 8),
            16,
            1,
            2,
            4,
            10,
            None,
            None,
            True,
            False,
        ),
        (
            1,
            1,
            1,
            1,
            8192,
            320,
            512,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_INTERLEAVED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            2,
            8,
            1,
            32,
            4,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            2,
            1,
            1,
            96,
            768,
            1024,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_DRAM_INTERLEAVED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 8),
            4,
            1,
            4,
            1,
            4,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            16,
            1,
            16,
            4096,
            64,
            96,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_HEIGHT_SHARDED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseProgramConfig",
            (2, 8),
            2,
            1,
            1,
            128,
            3,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            16,
            1,
            16,
            4096,
            96,
            64,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseProgramConfig",
            (2, 8),
            3,
            1,
            1,
            128,
            2,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            8192,
            320,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            2,
            1,
            8,
            32,
            8,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            8192,
            320,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            2,
            1,
            8,
            32,
            8,
            False,
            "GELU",
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            8192,
            1280,
            320,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            8,
            4,
            2,
            32,
            2,
            False,
            None,
            None,
            None,
        ),
        (
            2,
            1,
            1,
            1,
            32,
            1280,
            640,
            "BFLOAT16",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_INTERLEAVED",
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            (8, 8),
            1,
            2,
            1,
            2,
            1,
            None,
            None,
            True,
            True,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            320,
            640,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 5),
            2,
            1,
            4,
            8,
            4,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            640,
            640,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 5),
            4,
            1,
            4,
            8,
            4,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            640,
            2304,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            4,
            8,
            1,
            8,
            17,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            1024,
            96,
            1024,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_HEIGHT_SHARDED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            (8, 8),
            3,
            1,
            8,
            1,
            32,
            None,
            None,
            True,
            False,
        ),
        (
            1,
            1,
            1,
            1,
            1024,
            1024,
            96,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_HEIGHT_SHARDED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            (8, 8),
            32,
            1,
            3,
            1,
            3,
            None,
            None,
            True,
            False,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            768,
            640,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (4, 8),
            6,
            1,
            5,
            8,
            5,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            640,
            768,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            4,
            8,
            1,
            8,
            5,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            2,
            1,
            1,
            96,
            768,
            1536,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_DRAM_INTERLEAVED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 8),
            4,
            1,
            6,
            1,
            6,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            16,
            1,
            16,
            1024,
            96,
            96,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_HEIGHT_SHARDED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseProgramConfig",
            (2, 8),
            3,
            1,
            1,
            32,
            3,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            16,
            1,
            16,
            1024,
            96,
            96,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseProgramConfig",
            (2, 8),
            3,
            1,
            1,
            32,
            3,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            640,
            2560,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            4,
            1,
            8,
            8,
            16,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            640,
            2560,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            4,
            1,
            8,
            8,
            16,
            False,
            "GELU",
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            2560,
            640,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            16,
            2,
            4,
            8,
            4,
            False,
            None,
            None,
            None,
        ),
        (
            2,
            1,
            1,
            1,
            32,
            1280,
            1280,
            "BFLOAT16",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_INTERLEAVED",
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            (8, 8),
            1,
            2,
            1,
            2,
            1,
            None,
            None,
            True,
            True,
        ),
        (
            1,
            1,
            1,
            1,
            512,
            640,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 5),
            4,
            1,
            4,
            2,
            8,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            512,
            1280,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 8),
            5,
            1,
            1,
            2,
            5,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            512,
            1280,
            3840,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 8),
            5,
            2,
            1,
            2,
            17,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            16,
            1,
            16,
            256,
            160,
            256,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_HEIGHT_SHARDED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseProgramConfig",
            (2, 8),
            5,
            1,
            1,
            8,
            8,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            16,
            1,
            16,
            256,
            256,
            160,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseProgramConfig",
            (2, 8),
            8,
            1,
            1,
            8,
            5,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            512,
            1280,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (5, 8),
            8,
            1,
            8,
            2,
            8,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            512,
            1280,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 8),
            5,
            2,
            1,
            2,
            5,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            2,
            1,
            1,
            96,
            768,
            2560,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_DRAM_INTERLEAVED",
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            (8, 8),
            1,
            3,
            2,
            6,
            2,
            None,
            None,
            True,
            True,
        ),
        (
            1,
            16,
            1,
            16,
            256,
            160,
            96,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_HEIGHT_SHARDED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseProgramConfig",
            (2, 8),
            5,
            1,
            1,
            8,
            3,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            16,
            1,
            16,
            256,
            96,
            160,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseProgramConfig",
            (2, 8),
            3,
            1,
            1,
            8,
            5,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            512,
            1280,
            5120,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 8),
            5,
            1,
            5,
            2,
            20,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            512,
            1280,
            5120,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 8),
            5,
            1,
            5,
            2,
            20,
            False,
            "GELU",
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            512,
            5120,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 8),
            20,
            1,
            5,
            2,
            5,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            128,
            1280,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (4, 8),
            5,
            1,
            1,
            1,
            5,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            128,
            1280,
            3840,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 4),
            5,
            1,
            1,
            1,
            17,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            16,
            1,
            16,
            64,
            160,
            64,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_HEIGHT_SHARDED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseProgramConfig",
            (2, 8),
            5,
            1,
            1,
            2,
            2,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            16,
            1,
            16,
            64,
            64,
            160,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_HEIGHT_SHARDED",
            "MatmulMultiCoreReuseProgramConfig",
            (2, 8),
            2,
            1,
            1,
            2,
            5,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            128,
            1280,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 4),
            5,
            1,
            5,
            1,
            5,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            128,
            1280,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 4),
            5,
            1,
            1,
            1,
            5,
            False,
            None,
            None,
            None,
        ),
        (
            2,
            8,
            2,
            8,
            96,
            160,
            96,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_DRAM_INTERLEAVED",
            "DEV_0_L1_INTERLEAVED",
            "",
            (1, 1),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            2,
            8,
            2,
            8,
            64,
            96,
            160,
            "BFLOAT8_B",
            "BFLOAT8_B",
            True,
            "DEV_0_L1_INTERLEAVED",
            "DEV_0_DRAM_INTERLEAVED",
            "",
            (1, 1),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            128,
            1280,
            5120,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 4),
            5,
            1,
            5,
            1,
            20,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            128,
            1280,
            5120,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 4),
            5,
            1,
            5,
            1,
            20,
            False,
            "GELU",
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            128,
            5120,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 4),
            20,
            1,
            5,
            1,
            5,
            False,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            128,
            2560,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (4, 8),
            10,
            1,
            1,
            1,
            5,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            512,
            2560,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 8),
            10,
            1,
            1,
            2,
            5,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            512,
            1920,
            1280,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 5),
            12,
            1,
            4,
            2,
            8,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            1920,
            640,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 5),
            12,
            1,
            4,
            8,
            4,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            1280,
            640,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 5),
            8,
            1,
            4,
            8,
            4,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            2048,
            960,
            640,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 5),
            6,
            1,
            4,
            8,
            4,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            8192,
            960,
            320,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 5),
            6,
            2,
            2,
            32,
            2,
            True,
            None,
            None,
            None,
        ),
        (
            1,
            1,
            1,
            1,
            8192,
            640,
            320,
            "BFLOAT8_B",
            "BFLOAT8_B",
            False,
            "DEV_0_L1_BLOCK_SHARDED",
            "DEV_0_L1_BLOCK_SHARDED",
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            (8, 5),
            4,
            2,
            2,
            32,
            2,
            True,
            None,
            None,
            None,
        ),
    ],
)
def test_matmul(
    device,
    W0,
    Z0,
    W1,
    Z1,
    M,
    K,
    N,
    input_dtype,
    output_dtype,
    bias,
    output_mem_config,
    input_mem_config,
    program_config_type,
    grid_size,
    in0_block_w,
    out_subblock_h,
    out_subblock_w,
    per_core_M,
    per_core_N,
    transpose_mcast,
    fused_activation,
    fuse_batch,
    mcast_in0,
):
    compute_grid_size = device.compute_with_storage_grid_size()

    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    in_0_shape = [W0, Z0, M, K]
    in_1_shape = [W1, Z1, K, N]
    in_2_shape = [1, 1, 1, K]

    in_0_torch = torch.randn(in_0_shape).bfloat16().float()
    in_1_torch = torch.randn(in_1_shape).bfloat16().float()
    bias = False  # TODO
    if bias:
        in_2_torch = torch.randn(in_2_shape)

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    block_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )
    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    in_0 = torch2tt_tensor(
        in_0_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=(
            ttnn.experimental.tensor.DataType.BFLOAT8_B
            if input_dtype == "BFLOAT8_B"
            else ttnn.experimental.tensor.DataType.BFLOAT16
        ),
    )
    in_1 = torch2tt_tensor(
        in_1_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
    )
    if bias:
        in_2 = torch2tt_tensor(
            in_2_torch,
            device,
            tt_memory_config=l1_interleaved_memory_config,
            tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
        )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    if input_mem_config == "DEV_0_L1_BLOCK_SHARDED":
        logical_grid = [grid_size[0], grid_size[1]] if transpose_mcast else [grid_size[1], grid_size[0]]
        in_0 = ttnn.interleaved_to_sharded(
            in_0,
            grid_size,
            [M * Z0 * W0 // logical_grid[0], K // logical_grid[1]],
            ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            (
                ttnn.experimental.tensor.ShardOrientation.COL_MAJOR
                if transpose_mcast
                else ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR
            ),
        )
    elif input_mem_config == "DEV_0_L1_HEIGHT_SHARDED":
        in_0 = ttnn.interleaved_to_sharded(
            in_0,
            grid_size,
            [round_up_to_tile_dim(M * Z0 * W0 // (grid_size[0] * grid_size[1])), round_up_to_tile_dim(K)],
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            (
                ttnn.experimental.tensor.ShardOrientation.COL_MAJOR
                if transpose_mcast
                else ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR
            ),
        )
    elif input_mem_config == "DEV_0_L1_INTERLEAVED":
        in_0 = ttnn.to_memory_config(in_0, l1_interleaved_memory_config)
    elif input_mem_config == "DEV_0_DRAM_INTERLEAVED":
        in_0 = ttnn.to_memory_config(in_0, dram_interleaved_memory_config)

    in_1 = ttnn.to_memory_config(in_1, dram_interleaved_memory_config)
    if fused_activation == "GELU":
        fused_activation = [ttnn.UnaryOpType.GELU, True]
    elif fused_activation == "RELU":
        fused_activation = [ttnn.UnaryOpType.RELU, True]
    else:
        assert fused_activation is None

    if program_config_type == "MatmulMultiCoreReuseMultiCastProgramConfig":
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=transpose_mcast,
            fused_activation=fused_activation,
        )
    elif program_config_type == "MatmulMultiCoreReuseMultiCast1DProgramConfig":
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fuse_batch=fuse_batch,
            fused_activation=fused_activation,
            mcast_in0=mcast_in0,
        )
    if output_mem_config == "DEV_0_L1_BLOCK_SHARDED":
        output_mem_config = block_sharded_memory_config
    elif output_mem_config == "DEV_0_L1_HEIGHT_SHARDED":
        output_mem_config = height_sharded_memory_config
    elif output_mem_config == "DEV_0_L1_INTERLEAVED":
        output_mem_config = l1_interleaved_memory_config
    elif output_mem_config == "DEV_0_DRAM_INTERLEAVED":
        output_mem_config = dram_interleaved_memory_config
    else:
        assert False

    if W1 > 1 or Z1 > 1:
        out = ttnn.matmul(
            in_0,
            in_1,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        out = ttnn.linear(
            in_0,
            in_1,
            bias=in_2 if bias else None,
            program_config=program_config,
            memory_config=output_mem_config,
            dtype=(
                ttnn.experimental.tensor.DataType.BFLOAT8_B
                if output_dtype == "BFLOAT8_B"
                else ttnn.experimental.tensor.DataType.BFLOAT16
            ),
            compute_kernel_config=compute_kernel_config,
        )

    out_torch = tt2torch_tensor(out)
    golden = torch.matmul(in_0_torch, in_1_torch)
    if bias:
        golden += in_2_torch
    pcc = 0.99

    if (
        (M == 8192 and K == 320 and N == 1280)
        or (M == 2048 and K == 640 and N == 2560)
        or (M == 512 and K == 1280 and N == 5120)
        or (M == 128 and K == 1280 and N == 5120)
    ):
        pcc = 0.85

    passing, output = comp_pcc(out_torch, golden, pcc=pcc)

    print(output)
    assert passing
