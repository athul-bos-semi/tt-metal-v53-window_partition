# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_allclose_and_pcc
from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)


def get_tensors(
    input_shape, other_shape, output_shape, require_input_grad, require_other_grad, is_1d, device, use_randint=True
):
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT

    # create tensors for forward
    if use_randint:
        input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype)
        other = torch.randint(-2, 3, other_shape, dtype=cpu_dtype)
        output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
    else:
        input = torch.rand(input_shape, dtype=cpu_dtype)
        other = torch.rand(other_shape, dtype=cpu_dtype)
        output = torch.rand(output_shape, dtype=cpu_dtype)

    tt_input = ttnn.Tensor(input, npu_dtype).pad_to_tile(float(1)).to(npu_layout).to(device)
    tt_other = ttnn.Tensor(other, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttnn.Tensor(output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    torch_input = input.reshape(-1) if is_1d else input
    torch_other = other.reshape(-1) if is_1d else other

    # tensors for backward
    output_grad = tt_output_grad = torch_output_grad = tt_input_grad = tt_other_grad = None
    if require_input_grad or require_other_grad:
        output_grad = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
        # tt_output_grad = ttnn.Tensor(output_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        tt_output_grad = ttnn.Tensor(output_grad, npu_dtype).pad_to_tile(float(-1)).to(npu_layout).to(device)
        torch_output_grad = output_grad[0][0][0][0] if is_1d else output_grad

        if require_input_grad:
            input_grad = torch.full(input_shape, float("nan"), dtype=cpu_dtype)
            tt_input_grad = ttnn.Tensor(input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

        if require_other_grad:
            other_grad = torch.full(other_shape, float("nan"), dtype=cpu_dtype)
            tt_other_grad = (
                ttnn.Tensor(
                    other_grad,
                    npu_dtype,
                )
                .pad_to_tile(float("nan"))
                .to(npu_layout)
                .to(device)
            )

    return (
        tt_input,
        tt_other,
        tt_output,
        tt_output_grad,
        tt_input_grad,
        tt_other_grad,
        torch_input,
        torch_other,
        torch_output_grad,
    )


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 1, 1, 10],  # test not mutiple of 32 case
        [1, 1, 1, 32],  # test single tile
        [1, 1, 1, 352],  # test multiple tiles
        [1, 1, 1, 323],  # test multiple tiles, not a multiple of 32
    ),
)
def test_moreh_matmul_1d(input_shape, device):
    torch.manual_seed(3072)
    # get tensors
    output_shape = [1, 1, 1, 1]
    tt_input, tt_other, _, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, input_shape, output_shape, False, False, True, device
    )

    # tt matmul
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_out = (
        ttnn.experimental.operations.primary.moreh_matmul(tt_input, tt_other)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # torch matmul
    torch_input = torch.reshape(torch_input, (torch_input.shape[-1],))
    torch_other = torch.reshape(torch_other, (torch_other.shape[-1],))
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_out[0][0][0][0], pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 1, 1, 10],  # test not mutiple of 32 case
        [1, 1, 1, 32],  # test single tile
        [1, 1, 1, 352],  # test multiple tiles
        [1, 1, 1, 323],  # test multiple tiles, not a multiple of 32
    ),
)
@pytest.mark.parametrize(
    "requires_grad",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
def test_moreh_matmul_1d_backward(input_shape, requires_grad, device):
    torch.manual_seed(3072)
    require_input_grad, require_other_grad = requires_grad
    output_shape = [1, 1, 1, 1]
    # get tensors
    (
        tt_input,
        tt_other,
        _,
        tt_output_grad,
        tt_input_grad,
        tt_other_grad,
        torch_input,
        torch_other,
        torch_output_grad,
    ) = get_tensors(input_shape, input_shape, output_shape, require_input_grad, require_other_grad, True, device)

    # torch matmul
    torch_out = torch.matmul(
        torch_input.requires_grad_(require_input_grad), torch_other.requires_grad_(require_other_grad)
    )
    torch_out.backward(torch_output_grad)

    # tt matmul backward
    ttnn.experimental.operations.primary.moreh_matmul_backward(
        tt_output_grad, tt_input, tt_other, (require_input_grad, require_other_grad), tt_input_grad, tt_other_grad
    )

    # test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    if require_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()

        passing, output_pcc = comp_allclose_and_pcc(
            torch_input.grad, ttcpu_input_grad.reshape(-1), pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"input_grad passing={passing}")
        logger.debug(f"input_grad pcc={output_pcc}")
        assert passing

    if require_other_grad:
        ttcpu_other_grad = tt_other_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()

        passing, output_pcc = comp_allclose_and_pcc(
            torch_other.grad, ttcpu_other_grad.reshape(-1), pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"other_grad passing={passing}")
        logger.debug(f"other_grad pcc={output_pcc}")
        assert passing


@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape
        ([3, 128, 96], [3, 4, 1, 96, 256], [3, 4, 3, 128, 256]),
        ([3, 3, 313, 511], [3, 3, 511, 765], [3, 3, 313, 765]),
        ([3, 1, 2, 1, 4, 1, 319, 95], [4, 2, 95, 470], [3, 1, 2, 1, 4, 2, 319, 470]),
        ([3, 2, 1, 470, 95], [2, 1, 3, 1, 2, 2, 95, 319], [2, 1, 3, 3, 2, 2, 470, 319]),
    ),
)
@pytest.mark.parametrize(
    "requires_grad",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
def test_moreh_matmul_backward(params, requires_grad, device):
    torch.manual_seed(3072)
    input_shape, other_shape, output_shape = params
    require_input_grad, require_other_grad = requires_grad

    # get tensors
    (
        tt_input,
        tt_other,
        _,
        tt_output_grad,
        tt_input_grad,
        tt_other_grad,
        torch_input,
        torch_other,
        torch_output_grad,
    ) = get_tensors(input_shape, other_shape, output_shape, require_input_grad, require_other_grad, False, device)

    # torch matmul
    torch_out = torch.matmul(
        torch_input.requires_grad_(require_input_grad), torch_other.requires_grad_(require_other_grad)
    )
    torch_out.backward(torch_output_grad)

    # tt matmul backward
    tt_input_grad, tt_other_grad = ttnn.experimental.operations.primary.moreh_matmul_backward(
        tt_output_grad, tt_input, tt_other, (require_input_grad, require_other_grad), tt_input_grad, tt_other_grad
    )

    # test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    if require_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, ttcpu_input_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"input_grad passing={passing}")
        logger.debug(f"input_grad pcc={output_pcc}")
        assert passing
    else:
        assert tt_input_grad is None

    if require_other_grad:
        ttcpu_other_grad = tt_other_grad.cpu().to(cpu_layout).unpad_from_tile(other_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(torch_other.grad, ttcpu_other_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"other_grad passing={passing}")
        logger.debug(f"other_grad pcc={output_pcc}")
        assert passing
    else:
        assert tt_other_grad is None


def moreh_matmul(params, has_output, compute_kernel_config, device):
    torch.manual_seed(3072)
    input_shape, other_shape, output_shape, transpose_input, transpose_other = params
    tt_input, tt_other, tt_output, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, other_shape, output_shape, False, False, False, device
    )
    if not has_output:
        tt_output = None

    torch_input = torch_input.transpose(-1, -2) if transpose_input else torch_input
    torch_other = torch_other.transpose(-1, -2) if transpose_other else torch_other

    # tt matmul
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output = ttnn.experimental.operations.primary.moreh_matmul(
        tt_input,
        tt_other,
        transpose_input=transpose_input,
        transpose_other=transpose_other,
        output=tt_output,
        compute_kernel_config=compute_kernel_config,
    )
    tt_output_cpu = tt_output.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # torch matmul
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    return passing


@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape, transpose input, other
        ([32, 32], [32, 32], [32, 32], False, False),  # single-core
        ([1024, 128], [128, 1024], [1024, 1024], False, False),  # multi-core
        ([128, 1024], [128, 1024], [1024, 1024], True, False),  # input transpose
        ([1024, 128], [1024, 128], [1024, 1024], False, True),  # other transpose
        ([128, 1024], [1024, 128], [1024, 1024], True, True),  # input, other transpose
        ([1020, 128], [128, 1024], [1020, 1024], False, False),  # input mask
        ([1024, 128], [128, 1020], [1024, 1020], False, False),  # other mask
        ([1020, 310], [310, 1020], [1020, 1020], False, False),  # input, other mask
        ([128, 1020], [128, 1024], [1020, 1024], True, False),  # input mask, transpose
        ([1024, 128], [1020, 128], [1024, 1020], False, True),  # other mask, transpose
        ([310, 1020], [1020, 310], [1020, 1020], True, True),  # input, other mask, transpose
        ([3, 1, 2, 1, 4, 1, 319, 95], [4, 2, 95, 470], [3, 1, 2, 1, 4, 2, 319, 470], False, False),  # batched matmul
        ([2, 319, 95], [2, 1, 3, 4, 1, 95, 470], [2, 1, 3, 4, 2, 319, 470], False, False),  # batched matmul
        ([3, 1, 2, 1, 4, 1, 95, 319], [4, 2, 95, 470], [3, 1, 2, 1, 4, 2, 319, 470], True, False),  # batched matmul
        ([2, 319, 95], [2, 1, 3, 4, 1, 470, 95], [2, 1, 3, 4, 2, 319, 470], False, True),  # batched matmul
        (
            [2, 3, 1, 2, 3, 2, 64, 64],
            [2, 1, 4, 2, 1, 2, 64, 64],
            [2, 3, 4, 2, 3, 2, 64, 64],
            False,
            False,
        ),  # batched matmul
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_matmul(params, compute_kernel_options, device):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    passing = moreh_matmul(params, True, compute_kernel_config, device)
    assert passing


@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape, transpose input, other
        ([32, 32], [32, 32], [32, 32], False, False),  # single-core
        ([3, 1, 2, 1, 4, 1, 95, 319], [4, 2, 95, 470], [3, 1, 2, 1, 4, 2, 319, 470], True, False),  # batched matmul
        ([2, 319, 95], [2, 1, 3, 4, 1, 470, 95], [2, 1, 3, 4, 2, 319, 470], False, True),  # batched matmul
        (
            [2, 3, 1, 2, 3, 2, 64, 64],
            [2, 1, 4, 2, 1, 2, 64, 64],
            [2, 3, 4, 2, 3, 2, 64, 64],
            False,
            False,
        ),  # batched matmul
    ),
)
def test_moreh_matmul_wo_output(params, device):
    passing = moreh_matmul(params, False, None, device)
    assert passing


@pytest.mark.parametrize(
    "params",
    (
        # input, weight, bias(1d or scalar), output
        ([32, 32], [32, 32], [32, 32], False, False),  # single-core
        (
            [2, 3, 1, 2, 3, 2, 64, 64],
            [2, 1, 4, 2, 1, 2, 64, 64],
            [2, 3, 4, 2, 3, 2, 64, 64],
            False,
            False,
        ),  # batched matmul
    ),
)
def test_moreh_matmul_enable_cache(params, device, use_program_cache):
    torch.manual_seed(3072)
    for i in range(4):
        # change input's transpose option
        if i % 2 == 1:
            param_list = list(params)
            param_list[3] = False if param_list[3] else True
            params = tuple(param_list)
        passing = moreh_matmul(params, True, None, device)
        assert passing
    assert device.num_program_cache_entries() == 2


@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape, transpose input, other
        ([32, 3200], [3200, 32], [32, 32], False, False),
        ([3100, 31], [3100, 31], [31, 31], True, False),
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_matmul_fp32_dest_acc(params, compute_kernel_options, device):
    torch.manual_seed(3072)
    input_shape, other_shape, output_shape, transpose_input, transpose_other = params
    tt_input, tt_other, tt_output, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, other_shape, output_shape, False, False, False, device, use_randint=False
    )

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    torch_input = torch_input.transpose(-1, -2) if transpose_input else torch_input
    torch_other = torch_other.transpose(-1, -2) if transpose_other else torch_other

    # tt matmul
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output = ttnn.experimental.operations.primary.moreh_matmul(
        tt_input,
        tt_other,
        transpose_input=transpose_input,
        transpose_other=transpose_other,
        output=tt_output,
        compute_kernel_config=compute_kernel_config,
    )
    tt_output_cpu = tt_output.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # torch matmul (float)
    torch_out = torch.matmul(torch_input.float(), torch_other.float())

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    diff = torch.abs(torch_out - tt_output_cpu)
    logger.debug(f"std={torch.std(diff)}")
    logger.debug(f"mean={diff.mean()}")
    logger.debug(f"topk(5) {torch.topk(diff.reshape(-1), 5)}")

    # TODO
    # assert passing


@pytest.mark.parametrize(
    "shape",
    ([32, 32],),
    ids=["32, 32"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_test1(shape, compute_kernel_options, device):
    passing = True
    npu_dtype = ttnn.uint8
    npu_layout = ttnn.TILE_LAYOUT
    torch_input = torch.randint(0, 256, shape)
    tt_input = ttnn.Tensor(torch_input, npu_dtype).to(npu_layout).to(device)

    torch_output = torch.zeros(shape)
    tt_output = ttnn.Tensor(torch_output, npu_dtype).to(npu_layout).to(device)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    ttnn.experimental.operations.primary.moreh_test(
        tt_input,
        output=tt_output,
        compute_kernel_config=compute_kernel_config,
    )
    tt_output_cpu = tt_output.cpu().to(cpu_layout).to_torch()

    torch.set_printoptions(threshold=1000000, linewidth=100000000, sci_mode=False)
    logger.debug(f"fp32_dest_acc_en = {compute_kernel_options}")
    # logger.debug(f"torch_input {torch_input}")
    # logger.debug(f"tt_output_cpu {tt_output_cpu}")
    logger.debug(f"tt_input {tt_input}")
    logger.debug(f"tt_output {tt_output}")
    passing = torch.equal(torch_input, tt_output_cpu)
    logger.debug(f"passing {passing}")
