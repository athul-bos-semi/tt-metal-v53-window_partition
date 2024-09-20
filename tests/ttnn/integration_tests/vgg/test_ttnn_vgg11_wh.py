# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from torchvision import models
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.vgg.tt import ttnn_vgg
from models.demos.wormhole.vgg.tt.vgg_preprocessing import custom_preprocessor
from PIL import Image
import torchvision.transforms as transforms


def imagenet_sample_input():
    path = "models/sample_data/ILSVRC2012_val_00048736.JPEG"
    im = Image.open(path)
    im = im.resize((224, 224))
    return transforms.ToTensor()(im).unsqueeze(0)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity", ((2, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),)
)
def test_vgg11(
    mesh_device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
):
    torch_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor_nchw = imagenet_sample_input().to(torch.bfloat16)
    torch_batched_tensor = torch_input_tensor_nchw.repeat(batch_size, 1, 1, 1)

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            convert_to_ttnn=lambda *_: True,
            device=mesh_device,
            custom_preprocessor=ttnn_vgg.custom_preprocessor,
        )

    model_config = {
        "MATH_FIDELITY": math_fidelity,
        "WEIGHTS_DTYPE": weight_dtype,
        "ACTIVATIONS_DTYPE": act_dtype,
    }

    torch_batched_tensor = torch_input_tensor_nchw.repeat(batch_size, 1, 1, 1)
    torch_input_tensor = torch.permute(torch_batched_tensor, (0, 2, 3, 1))
    tt_batched_input_tensor = ttnn.from_torch(
        torch_input_tensor, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=inputs_mesh_mapper
    )

    ttnn_output = ttnn_vgg.ttnn_vgg11(
        mesh_device,
        tt_batched_input_tensor,
        parameters,
        batch_size,
        model_config,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )
    torch_output_tensor = ttnn.to_torch(ttnn_output, mesh_composer=output_mesh_composer)
    golden_output = torch_model(torch_batched_tensor)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(
        (torch_output_tensor.squeeze(1)).squeeze(1), golden_output, pcc=0.99
    )
    logger.info(f"PCC: {pcc_msg}")
