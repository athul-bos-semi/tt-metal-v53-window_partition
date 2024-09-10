# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_pt_tt, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
@pytest.mark.parametrize("sizes", [[12, 1, 1, 1], [6, 1, 1, 1], [1, 24, 1, 1], [1, 3, 1, 1]])
def test_bw_repeat(input_shapes, sizes, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    pyt_y = in_data.repeat(sizes)
    grad_data, grad_tensor = data_gen_pt_tt(pyt_y.shape, device, True)

    opt_tensor = torch.zeros(input_shapes, dtype=torch.bfloat16)
    opt_tensor = ttnn.from_torch(
        opt_tensor,
        input_tensor.get_dtype(),
        layout=input_tensor.get_layout(),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # pages_before = ttnn._ttnn.reports.get_buffer_pages()
    # ttnn.repeat_bw(grad_tensor, input_tensor, sizes, output_tensor=opt_tensor)
    # print(len(pages_before),len(ttnn._ttnn.reports.get_buffer_pages() ) )
    # print(opt_tensor.shape)
    # assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages())

    # tt_output_tensor_on_device = [opt_tensor]
    tt_output_tensor_on_device = ttnn.repeat_bw(grad_tensor, input_tensor, sizes)
    print(tt_output_tensor_on_device[0].shape)

    golden_function = ttnn.get_golden_function(ttnn.repeat_bw)
    golden_tensor = golden_function(grad_data, in_data, sizes)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
