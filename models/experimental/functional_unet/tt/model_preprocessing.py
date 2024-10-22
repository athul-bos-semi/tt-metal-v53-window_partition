# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from typing import Literal

from ttnn.model_preprocessing import infer_ttnn_module_args

from models.experimental.functional_unet.tt import unet_shallow_torch


def create_unet_input_tensors(
    batch: int,
    groups: int,
    input_channels: int = 4,
    input_height: int = 1056,
    input_width: int = 160,
    channel_order: Literal["first", "last"] = "first",
    fold: bool = False,
    pad: bool = False,
    mesh_mapper=None,
):
    torch_input_tensor = torch.randn(batch, input_channels * groups, input_height, input_width)

    ttnn_input_tensor = (
        torch.nn.functional.pad(torch_input_tensor, (0, 0, 0, 0, 0, 16 - input_channels), mode="constant", value=0)
        if pad
        else torch_input_tensor
    )
    ttnn_input_tensor = ttnn_input_tensor if channel_order == "first" else ttnn_input_tensor.permute(0, 2, 3, 1)

    if fold:
        if channel_order == "first":
            raise RuntimeError("Cannot fold B x H x W when in channels first ordering")
        ttnn_input_tensor = ttnn_input_tensor.reshape(1, 1, batch * input_height * input_width, -1)

    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

    return torch_input_tensor, ttnn_input_tensor


def create_unet_model_parameters(model: unet_shallow_torch.UNet, input_tensor: torch.Tensor, groups: int):
    assert groups == 1, "Only groups=1 is supported for now"

    parameters = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.keys():
        parameters[key].module = getattr(model, key)

    parameters.p1["parallel_config_override"] = {
        "grid_size": (8, 8),
        "num_cores_nhw": 64,
    }
    parameters.p2["parallel_config_override"] = {
        "grid_size": (8, 8),
        "num_cores_nhw": 60,
    }
    parameters.p3["parallel_config_override"] = {
        "grid_size": (8, 8),
        "num_cores_nhw": 60,
    }
    parameters.p4["parallel_config_override"] = {
        "grid_size": (8, 7),
        "num_cores_nhw": 55,
    }

    parameters.c1["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 16 * 32}
    parameters.c1["use_split_reader"] = True
    parameters.c1["use_activation_double_buffer"] = True
    parameters.c1["input_channels_alignment"] = 16

    parameters.c1_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 16 * 32}
    parameters.c1_2["use_split_reader"] = True
    parameters.c1_2["use_activation_double_buffer"] = True

    parameters.c2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c2["use_split_reader"] = True
    parameters.c2["use_activation_double_buffer"] = True

    parameters.c2_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c2_2["use_activation_double_buffer"] = True
    parameters.c2_2["use_split_reader"] = True
    parameters.c2_2["use_activation_double_buffer"] = True

    parameters.c3["conv_blocking_and_parallelization_config_override"] = None
    parameters.c3["use_split_reader"] = True
    parameters.c3["use_activation_double_buffer"] = True
    parameters.c3_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c3_2["use_split_reader"] = True
    parameters.c3_2["use_activation_double_buffer"] = True

    parameters.c4["conv_blocking_and_parallelization_config_override"] = None
    parameters.c4["use_activation_double_buffer"] = True
    parameters.c4_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c4_2["use_activation_double_buffer"] = True

    parameters.bnc["conv_blocking_and_parallelization_config_override"] = None
    parameters.bnc["use_activation_double_buffer"] = True
    parameters.bnc_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.bnc_2["use_activation_double_buffer"] = True

    parameters.c5["conv_blocking_and_parallelization_config_override"] = None
    parameters.c5["use_activation_double_buffer"] = True
    parameters.c5_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c5_2["use_activation_double_buffer"] = True
    parameters.c5_3["conv_blocking_and_parallelization_config_override"] = None
    parameters.c5_3["use_activation_double_buffer"] = True

    parameters.c6["conv_blocking_and_parallelization_config_override"] = None
    parameters.c6["use_split_reader"] = True
    parameters.c6["use_activation_double_buffer"] = True
    parameters.c6_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c6_2["use_split_reader"] = True
    parameters.c6_2["use_activation_double_buffer"] = True
    parameters.c6_3["conv_blocking_and_parallelization_config_override"] = None
    parameters.c6_3["use_split_reader"] = True
    parameters.c6_3["use_activation_double_buffer"] = True

    parameters.c7["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 16 * 32}
    parameters.c7["use_activation_double_buffer"] = True
    parameters.c7["use_split_reader"] = True
    parameters.c7["input_channels_alignment"] = 16
    parameters.c7_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c7_2["use_split_reader"] = True
    parameters.c7_2["use_activation_double_buffer"] = True
    parameters.c7_3["conv_blocking_and_parallelization_config_override"] = None
    parameters.c7_3["use_split_reader"] = True
    parameters.c7_3["use_activation_double_buffer"] = True

    parameters.c8["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 16 * 32}
    parameters.c8["use_split_reader"] = True
    parameters.c8["use_activation_double_buffer"] = True
    parameters.c8_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 16 * 32}
    parameters.c8_2["use_activation_double_buffer"] = True
    parameters.c8_2["use_split_reader"] = True
    parameters.c8_3["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 16 * 32}
    parameters.c8_3["use_activation_double_buffer"] = True
    parameters.c8_3["use_split_reader"] = True

    parameters.output_layer["conv_blocking_and_parallelization_config_override"] = None

    return parameters
