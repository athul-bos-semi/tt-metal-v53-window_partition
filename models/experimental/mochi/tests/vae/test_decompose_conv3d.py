import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
import pytest
from loguru import logger
from models.experimental.mochi.common import compute_metrics


def conv3d_memory_and_flops(
    batch_size: int,
    in_channels: int,
    in_depth: int,
    in_height: int,
    in_width: int,
    out_channels: int,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
):
    """
    Calculate:
      1) Number of elements to store the input (input memory usage)
      2) Number of elements to store the kernel (kernel memory usage)
      3) Number of FLOPs for a forward pass of this 3D convolution layer

    Arguments match those of torch.nn.Conv3d (with the addition of batch_size
    and input spatial dimensions), i.e.:
      - in_channels, out_channels
      - kernel_size (int or tuple of 3 ints)
      - stride (int or tuple of 3 ints)
      - padding (int or tuple of 3 ints)
      - dilation (int or tuple of 3 ints)
      - groups
      - bias (bool)
    """
    import math

    # Helper to handle int or tuple for 3D args
    def _triple(x):
        if isinstance(x, int):
            return (x, x, x)
        return x

    k_d, k_h, k_w = _triple(kernel_size)
    s_d, s_h, s_w = _triple(stride)
    p_d, p_h, p_w = _triple(padding)
    d_d, d_h, d_w = _triple(dilation)

    # 1) Compute output spatial dimensions (using the standard PyTorch formula)
    #    out_dim = floor((in_dim + 2*pad - dilation*(kernel_size-1) - 1)/stride + 1)
    def out_size(in_size, k, s, p, d):
        return (in_size + 2 * p - d * (k - 1) - 1) // s + 1

    out_depth = out_size(in_depth, k_d, s_d, p_d, d_d)
    out_height = out_size(in_height, k_h, s_h, p_h, d_h)
    out_width = out_size(in_width, k_w, s_w, p_w, d_w)

    # 2) Memory for the input (in elements):
    #    We store (batch_size * in_channels * in_depth * in_height * in_width)
    input_memory = batch_size * in_channels * in_depth * in_height * in_width

    # 3) Memory for the kernel (in elements):
    #    Weights shape: (out_channels, in_channels/groups, k_d, k_h, k_w)
    #    + bias if bias=True (out_channels elements)
    kernel_elems_per_out_channel = (in_channels // groups) * k_d * k_h * k_w
    kernel_memory = out_channels * kernel_elems_per_out_channel
    if bias:
        kernel_memory += out_channels  # each output channel has one bias term

    # 4) FLOPs for the forward pass:
    #    For each output element, the convolution does:
    #       (in_channels/groups * k_d * k_h * k_w) multiply-accumulates.
    #    Count each multiply-add as 2 FLOPs.
    #    Number of output elements = (batch_size * out_channels * out_depth * out_height * out_width)
    output_elements = batch_size * out_channels * out_depth * out_height * out_width
    macs_per_output = (in_channels // groups) * k_d * k_h * k_w
    flops = output_elements * macs_per_output * 2

    return input_memory, kernel_memory, flops


if __name__ == "__main__":
    # Example usage:
    b = 2
    ic = 16
    id_ = 32
    ih = 64
    iw = 64
    oc = 32
    ksize = (3, 3, 3)
    stride = 2
    padding = 1
    dilation = 1
    groups = 1
    bias = True

    inp_mem, ker_mem, total_flops = conv3d_memory_and_flops(
        b, ic, id_, ih, iw, oc, ksize, stride, padding, dilation, groups, bias
    )

    print("Input Memory (elements):", inp_mem)
    print("Kernel Memory (elements):", ker_mem)
    print("FLOPs:", total_flops)


def decomposed_conv3d_torch(input, conv3d_module):
    """
    A decomposed conv3d that computes the 3D convolution by iterating
    over output depth indices and summing over kernel depth slices via 2D conv.

    Parameters:
      input         : Tensor of shape [N, C, D, H, W]
      conv3d_module : An nn.Conv3d module (with weight, bias, stride, padding, dilation, etc.)

    Returns:
      output        : Tensor of shape [N, out_channels, D_out, H_out, W_out]
    """
    # Extract conv3d parameters.
    weight = conv3d_module.weight  # [out_channels, in_channels, kD, kH, kW]
    bias = conv3d_module.bias
    stride_d, stride_h, stride_w = conv3d_module.stride
    pad_d, pad_h, pad_w = conv3d_module.padding
    dilation_d, dilation_h, dilation_w = conv3d_module.dilation
    kD, kH, kW = conv3d_module.kernel_size
    N, C, D, H, W = input.shape
    padding_mode = conv3d_module.padding_mode

    # Pad only along the depth dimension.
    # (Note: For H and W we rely on F.conv2d’s internal padding.)
    # F.pad takes padding in the order: (pad_left_W, pad_right_W, pad_top_H, pad_bottom_H, pad_front_D, pad_back_D)
    if padding_mode == "zeros":
        input_padded = F.pad(input, (0, 0, 0, 0, pad_d, pad_d))
    elif padding_mode == "replicate":
        input_padded = F.pad(input, (0, 0, 0, 0, pad_d, pad_d), mode="replicate")
    else:
        raise ValueError(f"Unsupported padding mode: {padding_mode}")

    # Compute effective kernel sizes and output dimensions.
    eff_kD = dilation_d * (kD - 1) + 1
    D_out = (D + 2 * pad_d - eff_kD) // stride_d + 1
    eff_kH = dilation_h * (kH - 1) + 1
    H_out = (H + 2 * pad_h - eff_kH) // stride_h + 1
    eff_kW = dilation_w * (kW - 1) + 1
    W_out = (W + 2 * pad_w - eff_kW) // stride_w + 1

    # Allocate the output tensor.
    output = torch.zeros((N, conv3d_module.out_channels, D_out, H_out, W_out), dtype=input.dtype, device=input.device)

    # Loop over each output depth index.
    for d in range(D_out):
        # For each output depth position d, conv3d computes:
        #   input_depth_index = d * stride_d + t * dilation_d   for each kernel depth slice t.
        # We accumulate the contributions from each t.
        out_slice = 0
        for t in range(kD):
            depth_index = d * stride_d + t * dilation_d
            # Extract the 2D slice from the padded input at this depth.
            # Shape: [N, C, H, W]
            slice_2d = input_padded[:, :, depth_index, :, :]

            # Use conv2d to process the H/W dimensions with the corresponding kernel slice.
            # Note: We pass padding=(pad_h, pad_w) so that conv2d’s padding matches nn.Conv3d’s H/W padding.
            out2d = F.conv2d(
                slice_2d,
                weight[:, :, t, :, :],
                bias=None,
                stride=(stride_h, stride_w),
                padding=(pad_h, pad_w),
                dilation=(dilation_h, dilation_w),
            )
            # out2d has shape: [N, out_channels, H_out, W_out]
            out_slice = out_slice + out2d
        # Save the accumulated result into the output tensor at depth index d.
        output[:, :, d, :, :] = out_slice

    # Add bias (if any) once at the end.
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    return output


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
    ids=["variant0", "variant1", "variant2", "variant3", "variant4"],
)
def test_decomposed_conv3d_torch(input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    # Set a manual seed for reproducibility.
    torch.manual_seed(42)

    # Define input dimensions.
    N, C, D, H, W = input_shape

    # Create a random input tensor.
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)

    # Create a Conv3d module with chosen parameters.
    in_channels = C
    dilation = (1, 1, 1)
    conv3d_module = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
        padding_mode=padding_mode,
    )

    # Compute the output using PyTorch's built-in conv3d.
    output_builtin = conv3d_module(input_tensor)

    # Compute the output using the decomposed conv3d (based on conv2d).
    output_decomposed = decomposed_conv3d_torch(input_tensor, conv3d_module)

    pcc, mse, mae = compute_metrics(output_builtin, output_decomposed)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")
    assert pcc > 0.99, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"
    # Compare the two outputs.
    # assert torch.allclose(output_builtin, output_decomposed, atol=1e-5), f"Outputs do not match!\nBuilt-in:\n{output_builtin}\n\nDecomposed:\n{output_decomposed}"


def decomposed_conv3d_tt(device, input, conv3d_module):
    """
    A decomposed conv3d that computes the 3D convolution by iterating
    over output depth indices and summing over kernel depth slices via 2D conv.

    Parameters:
      input         : Tensor of shape [N, C, D, H, W]
      conv3d_module : An nn.Conv3d module (with weight, bias, stride, padding, dilation, etc.)

    Returns:
      output        : Tensor of shape [N, out_channels, D_out, H_out, W_out]
    """
    # Extract conv3d parameters.
    weight = conv3d_module.weight  # [out_channels, in_channels, kD, kH, kW]
    bias = conv3d_module.bias
    stride_d, stride_h, stride_w = conv3d_module.stride
    pad_d, pad_h, pad_w = conv3d_module.padding
    dilation_d, dilation_h, dilation_w = conv3d_module.dilation
    kD, kH, kW = conv3d_module.kernel_size
    N, C, D, H, W = input.shape
    out_channels = conv3d_module.out_channels
    padding_mode = conv3d_module.padding_mode
    # Pad only along the depth dimension.
    # (Note: For H and W we rely on F.conv2d’s internal padding.)
    # F.pad takes padding in the order: (pad_left_W, pad_right_W, pad_top_H, pad_bottom_H, pad_front_D, pad_back_D)
    # TODO: Pad and permute on device
    # Check padding_mode
    if padding_mode == "zeros":
        input_padded = F.pad(input, (0, 0, 0, 0, pad_d, pad_d))
    elif padding_mode == "replicate":
        input_padded = F.pad(input, (0, 0, 0, 0, pad_d, pad_d), mode="replicate")
    else:
        raise ValueError(f"Unsupported padding mode: {padding_mode}")

    # Reshape to get depth in upper dim, and collapse NHW to rows
    input_padded = input_padded.permute(2, 0, 3, 4, 1).reshape(1, D + pad_d * 2, N * H * W, C)
    tt_input = ttnn.from_torch(input_padded, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Slice weight on kD
    tt_weights = [ttnn.from_torch(weight[:, :, t, :, :], dtype=ttnn.bfloat16) for t in range(kD)]
    if bias is not None:
        tt_bias = ttnn.from_torch(
            bias.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
        )  # Applied at the end

    # Compute effective kernel sizes and output dimensions.
    eff_kD = dilation_d * (kD - 1) + 1
    D_out = (D + 2 * pad_d - eff_kD) // stride_d + 1
    eff_kH = dilation_h * (kH - 1) + 1
    H_out = (H + 2 * pad_h - eff_kH) // stride_h + 1
    eff_kW = dilation_w * (kW - 1) + 1
    W_out = (W + 2 * pad_w - eff_kW) // stride_w + 1

    # Loop over each output depth index.
    print(f"Iterating over {D_out} depth indices and {kD} kernel depth slices")
    out_tensors = []
    for d in range(D_out):
        # For each output depth position d, conv3d computes:
        #   input_depth_index = d * stride_d + t * dilation_d   for each kernel depth slice t.
        # We accumulate the contributions from each t.

        for t in range(kD):
            depth_index = d * stride_d + t * dilation_d
            # Extract the 2D slice from the padded input at this depth.
            # Shape: [N, C, H, W]
            slice_2d = tt_input[:, depth_index : depth_index + 1]

            # Use conv2d to process the H/W dimensions with the corresponding kernel slice.
            [out2d, [out_height, out_width], [_, _]] = ttnn.conv2d(
                input_tensor=slice_2d,
                weight_tensor=tt_weights[t],
                in_channels=C,
                out_channels=out_channels,
                device=device,
                bias_tensor=None,
                kernel_size=(kH, kW),
                stride=(stride_h, stride_w),
                padding=(pad_h, pad_w),
                dilation=(dilation_h, dilation_w),
                batch_size=N,
                input_height=H,
                input_width=W,
                groups=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            # out2d has shape: [N, out_channels, H_out, W_out]
            if t == 0:
                out_slice = ttnn.typecast(out2d, ttnn.float32)
            else:
                out_slice = out_slice + ttnn.typecast(out2d, ttnn.float32)

        # Save the accumulated result into the output tensor at depth index d.
        out_tensors.append(out_slice)

    # TODO: Optimize concat
    # Concat can't handle large lists of tensors. Batch up concats on dim1
    batched_outputs = []
    for i in range(0, len(out_tensors), 32):
        batched_outputs.append(ttnn.concat(out_tensors[i : i + 32], dim=1))
        for used_tensor in out_tensors[i : i + 32]:
            ttnn.deallocate(used_tensor)
    output = ttnn.concat(batched_outputs, dim=1)

    if bias is not None:
        # inplace add to avoid OOM
        ttnn.add(output, tt_bias, output_tensor=output)
        # output = output + tt_bias

    tt_output_tensor = ttnn.from_device(output)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    torch_output_tensor = torch_output_tensor.reshape(D_out, N, H_out, W_out, out_channels).permute(1, 4, 0, 2, 3)

    return torch_output_tensor


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
    ids=["variant0", "variant1", "variant2", "variant3", "variant4"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_decomposed_conv3d_tt(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, use_program_cache
):
    device.enable_async(True)
    # Set a manual seed for reproducibility.
    torch.manual_seed(42)
    required_pcc = 0.98  # TODO: tighten up

    # Define input dimensions.
    N, C, D, H, W = input_shape

    input_datums, kernel_datums, conv_flops = conv3d_memory_and_flops(
        *input_shape, out_channels, kernel_size, stride, padding
    )
    bytes_per_datum = 2
    input_memory = input_datums * bytes_per_datum
    kernel_memory = kernel_datums * bytes_per_datum

    def format_memory(bytes):
        if bytes < 1024 * 1024:  # Less than 1MB
            return f"{bytes/1024:.2f} KB"
        elif bytes < 1024 * 1024 * 1024:  # Less than 1GB
            return f"{bytes/(1024*1024):.2f} MB"
        else:
            return f"{bytes/(1024*1024*1024):.2f} GB"

    print(
        f"Input memory: {format_memory(input_memory)}, kernel memory: {format_memory(kernel_memory)}, flops: {conv_flops/1e9:.2f} GFLOPS"
    )
    mem_bw = 200  # GB/s
    chip_flops = 4096 // 2 * 64 * 1e9 / 1e12  # HiFi2 TFlops
    print(f"Memory-bound time: {(input_memory + kernel_memory) / 1e9 / mem_bw} seconds")
    print(f"FLOPS-bound time: {conv_flops / 1e12 / chip_flops} seconds")
    return

    # Create a random input tensor.
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)

    # Create a Conv3d module with chosen parameters.
    in_channels = C
    dilation = (1, 1, 1)
    conv3d_module = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
        padding_mode=padding_mode,
    )

    # Compute the output using PyTorch's built-in conv3d.
    import time

    start = time.perf_counter()
    output_builtin = conv3d_module(input_tensor)
    end = time.perf_counter()
    logger.info(f"Built-in latency: {end - start} seconds")

    # Compute the output using the decomposed conv3d (based on conv2d).
    output_decomposed = decomposed_conv3d_tt(device, input_tensor, conv3d_module)

    pcc, mse, mae = compute_metrics(output_builtin, output_decomposed)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")
    assert pcc > required_pcc, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"

    # start = time.perf_counter()
    # output_decomposed = decomposed_conv3d_tt(device, input_tensor, conv3d_module)
    # end = time.perf_counter()
    # logger.info(f"Compiled decomposed latency: {end - start} seconds")
