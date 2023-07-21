import sys
import pytest

from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

import tt_lib as ttl

from tt_lib.utils import _nearest_32
from python_api_testing.models.utility_functions import comp_pcc

TILE_HEIGHT = TILE_WIDTH = 32


def shape_padded(shape):
    return [ shape[0], shape[1], _nearest_32(shape[2]), _nearest_32(shape[3]) ]

@pytest.mark.parametrize(
    "act_shape",
    ((  [1, 7, 7, 2048],
     (  [1, 1, 32, 64]))),
    ids=[   'resnet50_unpadded',
            'tile_divisible' ]
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16",],
)
def test_run_average_pool(act_shape, dtype):

    batch_size, _, _, channels = act_shape

    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    torch.manual_seed(0)

    act = torch.randn(act_shape, dtype=torch.bfloat16).float()
    ttact = ttl.tensor.Tensor(act.to(torch.bfloat16))
    act_shape_padded = shape_padded(act_shape)
    if act_shape != act_shape_padded:
        ttact = ttact.pad_to_tile(0.)
    ttact = ttact.to(device)

    out = ttl.tensor.average_pool_2d(ttact)

    out = out.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    out_shape = [ batch_size, 1, 1, channels ]
    out_shape_padded = shape_padded(out_shape)
    if out_shape != out_shape_padded:
        out = out.unpad_from_tile(out_shape)

    out_pytorch = torch.tensor(out.data()).reshape(out_shape)

    ttl.device.CloseDevice(device)

    ## reference
    act_channels_first = torch.permute(act, (0, 3, 1, 2)) # Torch operates on channels-first tensors
    golden_pytorch = torch.nn.AdaptiveAvgPool2d((1, 1))(act_channels_first)

    ## test for equivalance
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
    print(f'Passing PCC = {passing_pcc}')
    print(f'Output PCC = {output_pcc}')

    assert(passing_pcc)
