# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
import os
import ttnn
import pytest

from models.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)

from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_downsample_2d_new_conv import downsample_2d
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    get_default_compute_config,
    preprocess_and_push_input_to_device,
    post_process_output_and_move_to_host,
)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "block_index, hidden_states, shard_end_core, shard_shape",
    [
        (0, [2, 320, 64, 64], (4, 7), (1024, 64)),
        (1, [2, 640, 32, 32], (4, 7), (256, 128)),
        (2, [2, 1280, 16, 16], (7, 7), (64, 160)),
    ],
)
def test_downblock_512x512(reset_seeds, device, block_index, hidden_states, shard_end_core, shard_shape):
    # Initialize PyTorch component
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    torch_downsample = pipe.unet.down_blocks[block_index].downsamplers[0]

    # Initialize ttnn component
    reader_patterns_cache = {}
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.down_blocks[block_index].downsamplers[0]
    N, _, H, W = hidden_states
    compute_kernel_config = get_default_compute_config(device)

    ttnn_downsample = downsample_2d(device, parameters, reader_patterns_cache, N, H, W, compute_kernel_config)

    # Prepare inputs
    in_channels = hidden_states[1]
    out_channels = in_channels
    input_shape = hidden_states
    hidden_states = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)

    # Run PyTorch component
    torch_output = torch_downsample(hidden_states)

    # Prepare inputs for ttnn component
    hidden_states = preprocess_and_push_input_to_device(
        device,
        hidden_states,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(shard_end_core[0], shard_end_core[1]),
                        ),
                    }
                ),
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    # Run ttnn component
    output = ttnn_downsample(
        in_channels=out_channels,
        out_channels=out_channels,
        hidden_states=hidden_states,
        use_conv=True,
    )

    # Compare outputs
    output = post_process_output_and_move_to_host(output, N, H // 2, W // 2, out_channels)
    assert_with_pcc(torch_output, output, 0.99)
