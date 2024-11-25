# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import torch.nn as nn
from models.demos.wormhole.lenet.tt import tt_lenet
from models.demos.wormhole.lenet import lenet_utils
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import is_wormhole_b0, skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size",
    [64],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_lenet_inference(mesh_device, batch_size, model_location_generator, reset_seeds):
    num_classes = 10
    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = (2 * batch_size) if mesh_device_flag else batch_size
    test_input, images, outputs = lenet_utils.get_test_data(batch_size)

    pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
    torch_lenet, state_dict = lenet_utils.load_torch_lenet(pt_model_path, num_classes)
    model = torch_lenet.float()
    torch_output = model(test_input)
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model, custom_preprocessor=lenet_utils.custom_preprocessor
        )

    parameters = lenet_utils.custom_preprocessor_device(parameters, device=mesh_device)

    x = test_input.permute(0, 2, 3, 1)
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)
    tt_output = tt_lenet.lenet(
        x,
        batch_size,
        mesh_device,
        parameters,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )
    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output, tt_output, 0.99)
