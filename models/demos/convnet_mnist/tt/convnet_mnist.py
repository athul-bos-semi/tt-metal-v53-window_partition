# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from torch import nn


def convnet_mnist(
    input_tensor,
    parameters,
    device,
):
    batch_size = input_tensor.shape[0]

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    x = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    [x, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.conv1.weight,
        in_channels=1,
        out_channels=32,
        device=device,
        bias_tensor=parameters.conv1.bias,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=batch_size,
        input_height=input_tensor.shape[1],
        input_width=input_tensor.shape[2],
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        debug=True,
        groups=1,
        return_output_dim=True,
        return_weights_and_bias=False,
    )
    x = ttnn.relu(x)

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=x.shape[-1],
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )

    x, [out_height, out_width] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.conv2.weight,
        in_channels=32,
        out_channels=64,
        device=device,
        bias_tensor=parameters.conv2.bias,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=batch_size,
        input_height=15,
        input_width=15,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
        return_output_dim=True,
        return_weights_and_bias=False,
    )
    x = ttnn.relu(x)

    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=x.shape[-1],
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

    x = ttnn.reshape(x, (batch_size, 6, 6, 64))
    x = ttnn.permute(x, (0, 3, 1, 2))

    x = ttnn.reshape(x, (batch_size, -1))

    x = ttnn.to_device(x, device)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    x = ttnn.linear(x, parameters.fc1.weight, bias=parameters.fc1.bias, activation="relu")

    x = ttnn.linear(x, parameters.fc2.weight, bias=parameters.fc2.bias)

    output = ttnn.softmax(x, dim=-1, numeric_stable=True)
    return output


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, device):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        weight = model.weight
        bias = model.bias
        while weight.dim() < 4:
            weight = weight.unsqueeze(0)
        while bias.dim() < 4:
            bias = bias.unsqueeze(0)
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.bfloat16)

    return parameters
