import torch
import torch.nn as nn

import ttnn

import pytest
from time import time

def tt_to_torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    return tt_output.to_torch()

def torch_to_tt_tensor_rm(py_tensor, device, shape=None, put_on_device=True):
    if shape is None:
        shape = list(py_tensor.size())
        while len(shape) < 4:
            shape.insert(0, 1)

    tt_tensor = ttnn.Tensor(py_tensor.reshape(shape), ttnn.bfloat16)
    if put_on_device:
        tt_tensor = tt_tensor.to(device)
    return tt_tensor

def torch_to_tt_tensor(py_tensor, device):
    shape = list(py_tensor.size())
    while len(shape) < 4:
        shape.insert(0, 1)

    tt_tensor = (
        ttnn.Tensor(py_tensor.reshape(shape), ttnn.bfloat16)
        .to(
            ttnn.TILE_LAYOUT
        )  # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(device)  # move TT Tensor from host to TT accelerator device (device is of type ttnn.device.Device)
    )

    return tt_tensor

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C).reshape(-1, window_size*window_size, C)
    return windows

def main(device):

    batch_size = 1
    image_height = 224
    image_width = 224
    input_channels = 3
    output_classes = 1000

    test_image = torch.randn(batch_size, image_height, image_width, input_channels, dtype=torch.bfloat16)
    golden_output = window_partition(test_image, 7)

    test_image = test_image.reshape(1,1,batch_size*image_height*image_width,input_channels)
    tt_image = torch_to_tt_tensor_rm(test_image, device)
    print(">>> TTNN Image Shape = ", tt_image.shape)

    output = ttnn.swin_window_partition([tt_image], 7, [image_height,image_width])
    output = tt_to_torch_tensor(output)
    # output = output.reshape(image_width,input_channels)
    print(">>> TTNN Output Shape = ", output.shape)

    if torch.equal(output,golden_output):
        print("Operation verified with Golden operation")
    else:
        print("Something went wrong somewhere!")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=2, l1_small_size=10240)
    main(device)
    ttnn.close_device(device)