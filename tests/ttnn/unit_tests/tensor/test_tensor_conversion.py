# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np

import ttnn

tt_dtype_to_torch_dtype = {
    ttnn.uint8: torch.uint8,
    ttnn.uint16: torch.int16,
    ttnn.uint32: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.float32: torch.float,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.float,
    ttnn.bfloat4_b: torch.float,
}

tt_dtype_to_np_dtype = {
    ttnn.uint8: np.ubyte,
    ttnn.uint16: np.int16,
    ttnn.uint32: np.int32,
    ttnn.int32: np.int32,
    ttnn.float32: np.float32,
    ttnn.bfloat8_b: np.float32,
    ttnn.bfloat4_b: np.float32,
}


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.uint8,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.int32,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
    ],
)
@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize("python_lib", [torch, np])
def test_tensor_conversion_with_tt_dtype(python_lib, shape, tt_dtype, device):
    torch.manual_seed(0)

    if python_lib == torch:
        dtype = tt_dtype_to_torch_dtype[tt_dtype]

        if dtype in {torch.uint8, torch.int16, torch.int32}:
            py_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype)
        else:
            py_tensor = torch.rand(shape, dtype=dtype)

        from torch import allclose

    elif python_lib == np:
        if tt_dtype == ttnn.bfloat16:
            pytest.skip("ttnn.bloat16 dtype is not supported yet for numpy tensors!")
        dtype = tt_dtype_to_np_dtype[tt_dtype]

        if dtype in {np.ubyte, np.int16, np.int32}:
            py_tensor = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, shape, dtype=dtype)
        else:
            py_tensor = np.random.random(shape).astype(dtype=dtype)

        from numpy import allclose

    tt_tensor = ttnn.Tensor(py_tensor, tt_dtype)
    if tt_dtype in {ttnn.bfloat8_b, ttnn.bfloat4_b}:
        assert tt_tensor.storage_type() == ttnn.StorageType.OWNED
        tt_tensor = tt_tensor.to(ttnn.TILE_LAYOUT)
    else:
        assert tt_tensor.storage_type() == ttnn.StorageType.BORROWED

    tt_tensor = tt_tensor.to(device)
    tt_tensor = tt_tensor.cpu()

    if tt_dtype in {ttnn.bfloat8_b, ttnn.bfloat4_b}:
        tt_tensor = tt_tensor.to(ttnn.ROW_MAJOR_LAYOUT)

    if python_lib == torch:
        py_tensor_after_round_trip = tt_tensor.to_torch()
    elif python_lib == np:
        py_tensor_after_round_trip = tt_tensor.to_numpy()

    assert py_tensor.dtype == py_tensor_after_round_trip.dtype
    assert py_tensor.shape == py_tensor_after_round_trip.shape

    allclose_kwargs = {}
    if tt_dtype == ttnn.bfloat8_b:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttnn.bfloat4_b:
        allclose_kwargs = dict(atol=0.2)

    passing = allclose(py_tensor, py_tensor_after_round_trip, **allclose_kwargs)
    assert passing


string_to_torch_dtype = {
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float,
}

string_to_np_dtype = {
    "uint8": np.ubyte,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
}


@pytest.mark.parametrize(
    "python_dtype_str",
    [
        "uint8",
        "int16",
        "int32",
        "int64",
        "bfloat16",
        "float16",
        "float32",
    ],
)
@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize("python_lib", [torch, np])
def test_tensor_conversion_with_python_dtype(python_lib, shape, python_dtype_str, device):
    torch.manual_seed(0)

    if python_lib == torch:
        dtype = string_to_torch_dtype[python_dtype_str]

        if dtype in {torch.uint8, torch.int16, torch.int32, torch.int64}:
            py_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype)
        else:
            py_tensor = torch.rand(shape, dtype=dtype)

        from torch import allclose

    elif python_lib == np:
        if python_dtype_str in ("bfloat16", "float16"):
            pytest.skip("{} dtype is not supported yet for numpy tensors!".format(python_dtype_str))
        dtype = string_to_np_dtype[python_dtype_str]

        if dtype in {np.ubyte, np.int16, np.int32, np.int64}:
            py_tensor = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, shape, dtype=dtype)
        else:
            py_tensor = np.random.random(shape).astype(dtype=dtype)

        from numpy import allclose

    tt_tensor = ttnn.Tensor(py_tensor)
    assert tt_tensor.storage_type() == ttnn.StorageType.BORROWED

    tt_tensor = tt_tensor.to(device)
    tt_tensor = tt_tensor.cpu()

    if python_lib == torch:
        py_tensor_after_round_trip = tt_tensor.to_torch()
    elif python_lib == np:
        py_tensor_after_round_trip = tt_tensor.to_numpy()

    if python_dtype_str in ("int64", "float16"):
        pytest.xfail(
            "{} dtype is incorrectly handled in ttnn tensors, so roundtrip tests are not working!".format(
                python_dtype_str
            )
        )

    assert py_tensor.dtype == py_tensor_after_round_trip.dtype
    assert py_tensor.shape == py_tensor_after_round_trip.shape

    allclose_kwargs = {}

    passing = allclose(py_tensor, py_tensor_after_round_trip, **allclose_kwargs)
    assert passing
