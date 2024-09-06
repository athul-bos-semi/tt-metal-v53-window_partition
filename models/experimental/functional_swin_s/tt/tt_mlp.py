# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from torch import nn

program_configs = {
    "linear_1_config_1": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=6,  # pcc drop to 0.80 for mlp sub_module when kept 12 why? check this
        per_core_M=8,
        per_core_N=12,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_1_config_2": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=6,
        per_core_M=2,
        per_core_N=24,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_1_config_4": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=96,
        per_core_M=1,
        per_core_N=96,
        transpose_mcast=False,
        fused_activation=None,
    ),
    "linear_2_config_1": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=3,  # pcc drop to 0.80 for mlp sub_module when kept 12 why? check this
        per_core_M=8,
        per_core_N=3,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_2_config_2": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=2,
        per_core_N=6,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_2_config_3": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=4,
        per_core_N=12,
        transpose_mcast=False,
        fused_activation=None,
    ),
    "linear_2_config_4": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=24,
        per_core_M=1,
        per_core_N=24,
        transpose_mcast=False,
        fused_activation=None,
    ),
}


class TtMLP:
    def __init__(
        self,
        hidden_channels,
        device,
        parameters,
        inplace=None,
        activation_layer=ttnn.relu,
        norm_layer=None,
    ):
        self.params = {} if inplace is None else {"inplace": inplace}
        self.device = device
        self.parameters = parameters
        self.norm_layer = norm_layer
        self.hidden_channels = hidden_channels
        self.activation_layer = activation_layer

    def __call__(self, x):
        for hidden_dim in self.hidden_channels[:-1]:
            if x.shape[-1] == 96:
                x = ttnn.to_memory_config(
                    x,
                    memory_config=ttnn.create_sharded_memory_config(
                        x.shape,
                        core_grid=ttnn.CoreGrid(y=8, x=8),
                        strategy=ttnn.ShardStrategy.HEIGHT,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                    dtype=ttnn.bfloat8_b,
                )

                x = ttnn.linear(
                    x,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    # core_grid=ttnn.CoreGrid(y=8, x=8),
                    program_config=program_configs["linear_1_config_1"],
                )
            elif x.shape[-1] == 192:
                x = ttnn.to_memory_config(
                    x,
                    memory_config=ttnn.create_sharded_memory_config(
                        x.shape,
                        core_grid=ttnn.CoreGrid(y=8, x=8),
                        strategy=ttnn.ShardStrategy.HEIGHT,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                    dtype=ttnn.bfloat8_b,
                )

                x = ttnn.linear(
                    x,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    # core_grid=ttnn.CoreGrid(y=8, x=8),
                    program_config=program_configs["linear_1_config_2"],
                )
            elif (
                x.shape[-1] == 384
            ):  # Not able to use block or height strategy to create_sharded_memory_config for 8x8 gird but works for 8x4, input [1,32,32,384] , throws error "(shard_shape[0] % constants::TILE_HEIGHT == 0 && shard_shape[1] % constants::TILE_WIDTH == 0)"
                # x = ttnn.to_memory_config(
                #     x,
                #     memory_config=ttnn.create_sharded_memory_config(
                #         x.shape,
                #         core_grid=ttnn.CoreGrid(y=8, x=8),
                #         strategy=ttnn.ShardStrategy.BLOCK,
                #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
                #     ),
                #     dtype=ttnn.bfloat8_b,
                # )

                x = ttnn.linear(
                    x,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    # memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    # dtype=ttnn.bfloat8_b,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                )
            elif (
                x.shape[-1] == 768
            ):  # unable to keep the MM in shard, input_a=[1,16,16,768] , input_b(weight)=[768,3072],  check with unit test
                # x = ttnn.to_memory_config(
                #     x,
                #     memory_config=ttnn.create_sharded_memory_config(
                #         x.shape,
                #         core_grid=ttnn.CoreGrid(y=8, x=8),
                #         strategy=ttnn.ShardStrategy.BLOCK,
                #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
                #     ),
                #     dtype=ttnn.bfloat8_b,
                # )
                # print("x",x.memory_config())
                # x = ttnn.linear(
                #     x,
                #     self.parameters[0].weight,
                #     bias=self.parameters[0].bias,
                #     memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                #     dtype=ttnn.bfloat8_b,
                #     # core_grid=ttnn.CoreGrid(y=8, x=8),
                #     program_config=program_configs["linear_1_config_4"],
                # )
                x = ttnn.linear(
                    x,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                )

            # x = ttnn.linear(x, self.parameters[0].weight, bias=self.parameters[0].bias)
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            if self.norm_layer is not None:
                x = ttnn.layer_norm(x, weight=self.parameters.norm_weight, bias=self.parameters.norm_bias)
            x = self.activation_layer(x)

        if x.shape[-1] == 384:
            x = ttnn.to_memory_config(
                x,
                memory_config=ttnn.create_sharded_memory_config(
                    x.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            x = ttnn.linear(
                x,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_2_config_1"],
            )
        elif x.shape[-1] == 768:
            x = ttnn.to_memory_config(
                x,
                memory_config=ttnn.create_sharded_memory_config(
                    x.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            x = ttnn.linear(
                x,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_2_config_2"],
            )
        elif x.shape[-1] == 1536:
            x = ttnn.to_memory_config(
                x,
                memory_config=ttnn.create_sharded_memory_config(
                    x.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )
            x = ttnn.linear(
                x,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                # core_grid=ttnn.CoreGrid(y=8, x=8),
                program_config=program_configs["linear_2_config_3"],
            )
        elif x.shape[-1] == 3072:  # not able to use shard MM , error num_blocks_y <= num_cores_y, should unit_test it.
            # x = ttnn.to_memory_config(
            #         x,
            #         memory_config=ttnn.create_sharded_memory_config(
            #             x.shape,
            #             core_grid=ttnn.CoreGrid(y=8, x=8),
            #             strategy=ttnn.ShardStrategy.BLOCK,
            #             orientation=ttnn.ShardOrientation.ROW_MAJOR,
            #         ),
            #         dtype=ttnn.bfloat8_b,
            #     )
            x = ttnn.linear(
                x,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                # memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                # program_config=program_configs["linear_2_config_4"],
            )
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        return x
