// SPDX-FileCopyrightText: © 2025 BOS
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/cpp/ttnn/operations/swin_ops/window_partition/device/multi_core_program_factory.hpp"
#include "ttnn/cpp/ttnn/operations/swin_ops/window_partition/device/window_partition_device_operation.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::swin_ops::detail {

tt_metal::operation::ProgramWithCallbacks windowpart_multi_core(
    const std::vector<Tensor> &input_tensors, const uint32_t window_size, std::vector<uint32_t> resolution, Tensor &output) {

    tt_metal::Program program = tt_metal::CreateProgram();
    tt_metal::Device *device = output.device();
    const tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto all_cores = input_tensors[0].shard_spec().value().grid;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    TT_FATAL(num_cores_x * num_cores_y == window_size, "Currently, Window Partition operation only works if all sticks in the first row of windows are in the first core, and so on");
    // num_cores_x * num_cores_y may not give you the total number of cores

    auto input_shard_spec = input_tensors[0].shard_spec().value();
    auto input_stick_size_real = input_shard_spec.shape[1] * input_tensors[0].element_size();

    // input CBs
    uint32_t cb_src_id = 0;
    auto shard_spec = input_tensors[0].shard_spec().value();
    auto input_num_units_per_shard_height = shard_spec.shape[0];
    auto input_num_units_per_shard_width = 1;
    auto num_input_units = input_num_units_per_shard_height * input_num_units_per_shard_width;
    uint32_t num_windows = num_input_units / (window_size * window_size);
    auto stick_size = round_up_to_mul32(input_stick_size_real);
    tt_metal::CircularBufferConfig input_cb_config =
        tt_metal::CircularBufferConfig(num_input_units * stick_size, {{cb_src_id, cb_data_format}})
            .set_page_size(cb_src_id, stick_size)
            .set_globally_allocated_address(*input_tensors[0].buffer());
    auto cb_src = tt_metal::CreateCircularBuffer(program, all_cores, input_cb_config);
    TT_FATAL(stick_size == input_stick_size_real, "Input Stick Size is not a multiple of 32, this condition has not been accounted for in the code");

    // output CB
    uint32_t cb_dst_id = 16;
    uint32_t num_output_sticks = input_tensors[0].get_legacy_shape()[-2];
    uint32_t num_windows_per_core = div_up(resolution[1], window_size);
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(num_output_sticks * stick_size, {{cb_dst_id, cb_data_format}})
            .set_page_size(cb_dst_id, stick_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    
    // auto window_size = window_size;
    uint32_t stride = ((num_windows_per_core - 1) * window_size);
    auto stride_forward_size = stride * input_tensors[0].element_size();
    uint32_t num_output_sticks_per_core = num_windows_per_core * window_size * window_size;
    uint32_t window_row_sticks_size = window_size * stick_size;

    std::vector<uint32_t> compile_time_args_0 = {
        cb_src_id,
        cb_dst_id,
        stick_size,
        window_row_sticks_size,
        stride_forward_size,
        window_size,
        num_windows,
        num_output_sticks_per_core
    };
    // std::vector<uint32_t> compile_time_args_1 = {

    // };

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/swin_ops/window_partition/device/kernels/dataflow/"
        "reader_inplace_writer_window_partition.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(compile_time_args_0));
    
    // tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
    //     program,
    //     "ttnn/cpp/ttnn/operations/swin_ops/window_partition/device/kernels/dataflow/"
    //     "reader_inplace_writer_window_partition.cpp",
    //     all_cores,
    //     tt_metal::ReaderDataMovementConfig(compile_time_args_1));

    auto override_runtime_args_callback = [cb_src, cb_output](
                                                    const void *operation,
                                                    Program &program,
                                                    const std::vector<Tensor> &input_tensors,
                                                    const std::vector<std::optional<const Tensor>> &,
                                                    const std::vector<Tensor> &output_tensors) {
        UpdateDynamicCircularBufferAddress(program, cb_src, *input_tensors[0].buffer());
        UpdateDynamicCircularBufferAddress(program, cb_output, *output_tensors[0].buffer());
    };

    return {std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::swin_ops::detail