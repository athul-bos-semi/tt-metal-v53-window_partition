# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from pathlib import Path

from models.demos.convnet_mnist.tt.convnet_mnist import convnet_mnist, custom_preprocessor
from models.demos.convnet_mnist import convnet_mnist_preprocessing
from models.demos.convnet_mnist.convnet_mnist_utils import get_test_data
from models.experimental.convnet_mnist.reference.convnet import ConvNet
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_convnet_mnist(reset_seeds, device, model_location_generator):
    state_dict = torch.load(model_location_generator("convnet_mnist.pt", model_subdir="ConvNetMNIST"))

    test_input, images, outputs = get_test_data(8)

    model = ConvNet()
    model.load_state_dict(state_dict)
    model.eval()

    torch_output = model(test_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, convert_to_ttnn=lambda *_: True, custom_preprocessor=custom_preprocessor
    )
    parameters = convnet_mnist_preprocessing.custom_preprocessor(parameters, device=device)

    ttnn_input = torch.permute(test_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ttnn_output = convnet_mnist(
        input_tensor=ttnn_input,
        device=device,
        parameters=parameters,
    )
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)
