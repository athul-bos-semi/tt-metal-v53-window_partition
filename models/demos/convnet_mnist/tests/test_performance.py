# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import time
from pathlib import Path

from torchvision import models
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.demos.convnet_mnist.tt.convnet_mnist import convnet_mnist, custom_preprocessor
from models.demos.convnet_mnist import convnet_mnist_preprocessing
from models.experimental.convnet_mnist.reference.convnet import ConvNet
from models.utility_functions import is_grayskull


def get_expected_times(convnet_mnist):
    return (15.0, 9.2)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity", ((1, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),)
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 32),
    ],
)
def test_convnet_mnist(
    device, input_shape, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator
):
    disable_persistent_kernel_cache()
    state_dict = torch.load(model_location_generator("convnet_mnist.pt", model_subdir="ConvNetMNIST"))

    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))

    model = ConvNet()
    model.load_state_dict(state_dict)
    model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, convert_to_ttnn=lambda *_: True, custom_preprocessor=custom_preprocessor
    )
    parameters = convnet_mnist_preprocessing.custom_preprocessor(parameters, device=device)

    durations = []
    for i in range(2):
        start = time.time()
        ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

        ttnn_output = convnet_mnist(
            input_tensor=ttnn_input,
            device=device,
            parameters=parameters,
        )
        output = ttnn.from_device(ttnn_output)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("convnet_mnist")
    prep_perf_report(
        model_name="convnet_mnist",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 105.710],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_convnet_mnist(batch_size, expected_perf):
    subdir = "ttnn_convnet_mnist"
    num_iterations = 1
    margin = 0.03
    expected_perf = 1646.94 if is_grayskull() else 1575.76

    command = f"pytest tests/ttnn/integration_tests/convnet_mnist/test_convnet_mnist.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"ttnn_functional_convnet_mnist{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
