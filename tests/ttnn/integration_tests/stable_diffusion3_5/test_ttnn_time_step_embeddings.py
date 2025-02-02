# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from models.experimental.functional_stable_diffusion3_5.reference.time_step_embeddings import TimestepEmbedding
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_time_step_embeddings import (
    ttnn_TimestepEmbedding as tt_module,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.utility_functions import skip_for_grayskull

torch.manual_seed(42)


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name):
        parameters = {}
        if isinstance(model, TimestepEmbedding):
            parameters["linear_1"] = {}
            parameters["linear_1"]["weight"] = preprocess_linear_weight(model.linear_1.weight, dtype=ttnn.bfloat8_b)
            parameters["linear_1"]["bias"] = preprocess_linear_bias(model.linear_1.bias, dtype=ttnn.bfloat8_b)
            parameters["linear_2"] = {}
            parameters["linear_2"]["weight"] = preprocess_linear_weight(model.linear_2.weight, dtype=ttnn.bfloat8_b)
            parameters["linear_2"]["bias"] = preprocess_linear_bias(model.linear_2.bias, dtype=ttnn.bfloat8_b)
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "init_inputs,fwd_input",
    [
        ((256, 1536, "silu", None, None, None, True), (2, 256)),
    ],
)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_step_embeddings(init_inputs, fwd_input, device):
    torch_sub_module = TimestepEmbedding(
        init_inputs[0], init_inputs[1], init_inputs[2], init_inputs[3], init_inputs[4], init_inputs[5], init_inputs[6]
    ).to(dtype=torch.bfloat16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_sub_module, device=device, custom_preprocessor=create_custom_preprocessor(device)
    )
    torch_input = torch.randn(fwd_input, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_sub_module = tt_module(parameters)
    tt_out = tt_sub_module(tt_input, device)
    torch_out = (torch_sub_module(torch_input)[0],)  # .unsqueeze(0).unsqueeze(0)
    tt_out_in_torch = ttnn.to_torch(tt_out)
    assert_with_pcc(torch_out, tt_out_in_torch, 0.9999)
