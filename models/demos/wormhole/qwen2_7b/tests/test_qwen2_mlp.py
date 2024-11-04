# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.wormhole.qwen2_7b.tt.qwen2_common import (
    load_safetensor_weights,
)
from models.demos.wormhole.qwen2_7b.tt.qwen2_mlp import TtQwen2MLP
from models.demos.wormhole.qwen2_7b.tt.model_config import TtModelArgs
from models.demos.wormhole.qwen2_7b.reference.model import FeedForward
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size",
    (
        1,
        8,
        256,
    ),
)
def test_qwen2_mlp_inference(device, batch_size, use_program_cache, reset_seeds):
    device.enable_async(False)
    dtype = ttnn.bfloat8_b
    model_args = TtModelArgs(device=device)
    state_dict = load_safetensor_weights(model_args.consolidated_weights_path)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("model.layers.0.mlp."))}

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = FeedForward(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtQwen2MLP(
        device=device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )
    torch_input = torch.randn(1, 1, batch_size, model_args.dim)
    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )

    logger.info("Compilation pass for Qwen2_MLP")
    tt_output = tt_model(tt_input)

    logger.info("Performance pass for Qwen2_MLP")
    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Qwen2_MLP Passed!")
    else:
        logger.warning("Qwen2_MLP Failed!")

    assert passing, f"Qwen2_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
