// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_program_factory.hpp"

namespace ttnn::operations::conv::conv3d::detail {

operation::ProgramWithCallbacks conv3d_factory(
    const Tensor& input_tensor,
    // const Tensor& weight_tensor,
    uint32_t output_channels,
    std::array<uint32_t, 3> kernel_size,
    std::array<uint32_t, 3> stride,
    std::array<uint32_t, 3> padding,
    std::string padding_mode,
    uint32_t groups,
    // const Tensor& bias_tensor,
    const Tensor& output_tensor) {
    Program program = CreateProgram();

    /*
    First implementation just performs vol2col on a single core.
    */

    auto input_tensor_shape = input_tensor.get_logical_shape();
    uint32_t N = input_tensor_shape[0];
    uint32_t T_in = input_tensor_shape[1];
    uint32_t H_in = input_tensor_shape[2];
    uint32_t W_in = input_tensor_shape[3];
    uint32_t C_in = input_tensor_shape[4];

    auto output_tensor_shape = output_tensor.get_logical_shape();
    uint32_t patch_size = output_tensor_shape[1];
    auto dtype = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());

    uint32_t T_out = T_in + 2 * padding[0] - (kernel_size[0] - 1);
    uint32_t H_out = H_in + 2 * padding[1] - (kernel_size[1] - 1);
    uint32_t W_out = W_in + 2 * padding[2] - (kernel_size[2] - 1);
    uint32_t C_out = output_channels;

    bool is_padding_zeros = padding_mode == "zeros";

    auto cb_vol2col_id = tt::CBIndex::c_0;

    uint32_t in_row_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t out_row_size_bytes = output_tensor.buffer()->aligned_page_size();

    tt::log_info("Input tensor shape: N={}, T={}, H={}, W={}, C={}", N, T_in, H_in, W_in, C_in);
    tt::log_info("Output tensor shape: T={}, H={}, W={}, C={}", T_out, H_out, W_out, C_out);
    tt::log_info("Kernel size: {}x{}x{}", kernel_size[0], kernel_size[1], kernel_size[2]);
    tt::log_info("Stride: {}x{}x{}", stride[0], stride[1], stride[2]);
    tt::log_info("Padding: {}x{}x{}", padding[0], padding[1], padding[2]);
    tt::log_info("Groups: {}", groups);
    tt::log_info("Patch size: {}", patch_size);
    tt::log_info("Input row size (bytes): {}", in_row_size_bytes);
    tt::log_info("Output row size (bytes): {}", out_row_size_bytes);
    tt::log_info("Data type: {}", dtype);
    tt::log_info("Circular buffer ID: {}", cb_vol2col_id);

    std::vector<uint32_t> reader_compile_time_args = {
        N,                   // 0
        T_in,                // 1
        H_in,                // 2
        W_in,                // 3
        C_in,                // 4
        padding[0],          // 5: padding_t
        padding[1],          // 6: padding_h
        padding[2],          // 7: padding_w
        kernel_size[0],      // 8: kernel_size_t
        kernel_size[1],      // 9: kernel_size_h
        kernel_size[2],      // 10: kernel_size_w
        T_out,               // 11
        H_out,               // 12
        W_out,               // 13
        C_out,               // 14
        cb_vol2col_id,       // 15: cb_in
        in_row_size_bytes,   // 16
        out_row_size_bytes,  // 17
        is_padding_zeros     // 18
    };

    auto core_grid = CoreRange({0, 0}, {0, 0});
    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv3d/device/kernels/reader_vol2im.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Patch construction CB
    auto c_vol2col_config =
        CircularBufferConfig(1 * patch_size, {{cb_vol2col_id, dtype}}).set_page_size(cb_vol2col_id, patch_size);
    auto c_vol2col_res = CreateCircularBuffer(program, core_grid, c_vol2col_config);

    CoreCoord core = {0, 0};
    SetRuntimeArgs(
        program, reader_kernels_id, core, {input_tensor.buffer()->address(), output_tensor.buffer()->address()});

    auto override_runtime_arguments_callback =
        [](const void* operation,
           Program& program,
           const std::vector<Tensor>& input_tensors,
           const std::vector<std::optional<const Tensor>>& optional_input_tensors,
           const std::vector<Tensor>& output_tensors) {};
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::conv::conv3d::detail
