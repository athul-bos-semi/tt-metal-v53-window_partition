# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import time
import os

from torchvision import models
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.yolov4.ttnn.yolov4 import TtYOLOv4
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import is_grayskull
from models.utility_functions import (
    profiler,
)


def get_expected_compile_time_sec():
    return 60


def get_expected_inference_time_sec():
    return 0.237


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 320, 320, 3),
    ],
)
def test_yolov4(
    device,
    input_shape,
    model_location_generator,
):
    disable_persistent_kernel_cache()
    profiler.clear()
    model_path = model_location_generator("models", model_subdir="Yolo")
    batch_size = input_shape[0]

    if model_path == "models":
        if not os.path.exists("tests/ttnn/integration_tests/yolov4/yolov4.pth"):  # check if yolov4.th is availble
            os.system(
                "tests/ttnn/integration_tests/yolov4/yolov4_weights_download.sh"
            )  # execute the yolov4_weights_download.sh file

        weights_pth = "tests/ttnn/integration_tests/yolov4/yolov4.pth"
    else:
        weights_pth = str(model_path / "yolov4.pth")
    ttnn_model = TtYOLOv4(device, weights_pth)

    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    out1, out2, out3 = ttnn_model(ttnn_input)
    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    iterations = 16
    outputs = []
    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        out1, out2, out3 = ttnn_model(ttnn_input)
        outputs.append(ttnn.from_device(out1, blocking=False))
        outputs.append(ttnn.from_device(out2, blocking=False))
        outputs.append(ttnn.from_device(out3, blocking=False))
        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Model compilation took {compile_time:.1f} s")
    logger.info(f"Inference time on last iterations was completed in {(inference_time * 1000.0):.2f} ms")
    logger.info(
        f"Mean inference time for {batch_size} (batch) images was {(mean_inference_time * 1000.0):.2f} ms ({batch_size / mean_inference_time:.2f} fps)"
    )

    expected_compile_time = get_expected_compile_time_sec()
    expected_inference_time = get_expected_inference_time_sec()

    prep_perf_report(
        model_name="yolov4",
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
    "batch_size, model_name",
    [
        (1, "yolov4"),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov4(batch_size, model_name):
    subdir = model_name
    num_iterations = 1
    margin = 0.03

    expected_perf = 234
    command = f"pytest tests/ttnn/integration_tests/yolov4/test_ttnn_yolov4.py"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
