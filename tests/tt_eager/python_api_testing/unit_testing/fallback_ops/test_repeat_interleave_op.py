# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib as ttl
import tt_lib.fallback_ops
from models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from loguru import logger
import pytest


@pytest.mark.parametrize("input_shape", [torch.Size([1, 1, 32, 32]), torch.Size([2, 3, 5, 6])])
@pytest.mark.parametrize("repeats", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("on_device", [False, True])
def test_repeat_interleave_fallback(input_shape, repeats, dim, on_device, device):
    torch.manual_seed(1234)

    x = torch.randn(input_shape).bfloat16().float()
    pt_out = torch.repeat_interleave(x, repeats, dim)

    # Test on host RM
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.fallback_ops.repeat_interleave(t0, repeats, dim)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
