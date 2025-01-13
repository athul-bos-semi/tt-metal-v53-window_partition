# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.ttnn_resnet.tests.perf_e2e_resnet50 import run_perf_resnet
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, enable_async_mode, expected_inference_time, expected_compile_time",
    ((16, True, 0.0100, 60),),
    indirect=["enable_async_mode"],
)
def test_perf(
    mesh_device,
    use_program_cache,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    enable_async_mode,
    model_location_generator,
):
    mode = "async" if enable_async_mode else "sync"
    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50_{mode}",
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1500000}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, enable_async_mode, expected_inference_time, expected_compile_time",
    ((16, True, 0.0057, 60),),
    indirect=["enable_async_mode"],
)
def test_perf_trace(
    mesh_device,
    use_program_cache,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    enable_async_mode,
    model_location_generator,
):
    mode = "async" if enable_async_mode else "sync"
    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50_trace_{mode}",
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, enable_async_mode, expected_inference_time, expected_compile_time",
    ((16, True, 0.0105, 60),),
    indirect=["enable_async_mode"],
)
def test_perf_2cqs(
    mesh_device,
    use_program_cache,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    enable_async_mode,
    model_location_generator,
):
    mode = "async" if enable_async_mode else "sync"
    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50_2cqs_{mode}",
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1332224}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size, enable_async_mode, expected_inference_time, expected_compile_time",
    ((16, True, 0.0043, 60),),
    indirect=["enable_async_mode"],
)
def test_perf_trace_2cqs(
    mesh_device,
    use_program_cache,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    enable_async_mode,
    model_location_generator,
):
    mode = "async" if enable_async_mode else "sync"
    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50_trace_2cqs_{mode}",
        model_location_generator,
    )
