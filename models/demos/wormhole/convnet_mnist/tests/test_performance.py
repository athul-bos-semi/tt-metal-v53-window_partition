# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import time
from pathlib import Path

from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.demos.wormhole.convnet_mnist.tt.convnet_mnist import (
    convnet_mnist,
    custom_preprocessor,
)
from models.demos.wormhole.convnet_mnist import convnet_mnist_preprocessing
from models.experimental.convnet_mnist.reference.convnet import ConvNet
from models.utility_functions import is_wormhole_b0, skip_for_grayskull


def get_expected_times(convnet_mnist):
    return (15.0, 9.2)


@skip_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [
        128,
    ],
)
def test_convnet_mnist(
    mesh_device,
    batch_size,
    reset_seeds,
    model_location_generator,
):
    disable_persistent_kernel_cache()

    state_dict = torch.load(model_location_generator("convnet_mnist.pt", model_subdir="ConvNetMNIST"))

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 2 * batch_size if mesh_device_flag else batch_size

    input_tensor = torch.randn([batch_size, 1, 32, 32], dtype=torch.bfloat16)
    batch_size = input_tensor.shape[0]
    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))

    model = ConvNet()
    model.load_state_dict(state_dict)
    model.eval()

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model, convert_to_ttnn=lambda *_: True, custom_preprocessor=custom_preprocessor
        )
        parameters = convnet_mnist_preprocessing.custom_preprocessor(parameters, device=mesh_device)

    durations = []
    for i in range(2):
        start = time.time()
        ttnn_input = ttnn.from_torch(
            input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=inputs_mesh_mapper
        )

        ttnn_output = convnet_mnist(
            input_tensor=ttnn_input,
            device=mesh_device,
            parameters=parameters,
            mesh_mapper=inputs_mesh_mapper,
            mesh_composer=output_mesh_composer,
        )

        output = ttnn.to_torch(ttnn_output, mesh_composer=output_mesh_composer)
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
    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"
    logger.info("Exit ConvNet Mnist perf test")


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [256, 75680.48],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_convnet_mnist(batch_size, expected_perf):
    subdir = "ttnn_convnet_mnist"
    num_iterations = 1
    margin = 0.03

    command = f"pytest tests/ttnn/integration_tests/convnet_mnist/test_convnet_mnist_wh.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"ttnn_convnet_mnist_wh_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
