// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/fill_pad/device/fill_pad_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_log.h>

bool is_power_of_two_at_least_32(uint32_t value) { return value >= 32 && (value & (value - 1)) == 0; }

using namespace tt;

std::map<DataType, uint32_t> data_type_to_size = {
    {DataType::BFLOAT16, 2},
    {DataType::FLOAT32, 4},
    {DataType::UINT32, 4},
    {DataType::UINT8, 1},
};

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks fill_pad_multi_core(const Tensor& input_tensor, float fill_value) {
    tt::tt_metal::IDevice* device = input_tensor.device();
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    tt::tt_metal::Buffer* tens_buffer = input_tensor.buffer();
    TT_ASSERT(tens_buffer != nullptr, "Input buffer should be allocated on device!");

    uint32_t input_element_size_bytes = data_type_to_size[input_tensor.get_dtype()];
    uint32_t cb_page_size = input_element_size_bytes * tt::constants::FACE_HEIGHT + sizeof(uint16_t);
    uint32_t height = input_tensor.get_logical_shape()[-2];
    uint32_t width = input_tensor.get_logical_shape()[-1];

    uint32_t problem_size = input_tensor.get_logical_shape()[-3];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, problem_size);
    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_page_size * 2, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    bool src_is_dram = tens_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    // pack bf16 vals
    uint32_t packed_fill_value = (std::uint32_t)fill_value;
    if (input_tensor.get_dtype() == DataType::BFLOAT16) {
        packed_fill_value = pack_two_bfloat16_into_uint32({bfloat16(fill_value), bfloat16(fill_value)});
    }

    uint32_t padded_height = tt::div_up(height, tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    uint32_t padded_width = tt::div_up(width, tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    uint32_t tiles_per_2d_tensor =
        padded_height / tt::constants::TILE_HEIGHT * padded_width / tt::constants::TILE_HEIGHT;
    uint32_t tiles_per_tile_row = padded_width / tt::constants::TILE_HEIGHT;

    // create kernel
    // reader compile time args
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)packed_fill_value,
        (std::uint32_t)input_element_size_bytes,
        (std::uint32_t)height,
        (std::uint32_t)width,
        (std::uint32_t)padded_height,
        (std::uint32_t)padded_width,
        (std::uint32_t)tiles_per_2d_tensor,
        (std::uint32_t)tiles_per_tile_row,
        (std::uint32_t)tt::constants::TILE_HEIGHT,
        (std::uint32_t)tt::constants::FACE_HEIGHT};

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_writer.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));  // writer only for in-place operation

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    std::vector<uint32_t> writer_runtime_args = {
        (std::uint32_t)tens_buffer->address(), (std::uint32_t)cb_page_size, (std::uint32_t)0, (std::uint32_t)0};

    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t local_num_2d_tensors = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;
        // Writer
        {
            writer_runtime_args[2] = tile_offset;
            writer_runtime_args[3] = local_num_2d_tensors;
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }

        tile_offset += local_num_2d_tensors * tiles_per_2d_tensor;
    }

    auto override_runtime_args_callback = [writer_kernel_id, cores](
                                              const Program& program,
                                              const std::vector<Buffer*>& input_buffers,
                                              const std::vector<Buffer*>& output_buffers) {
        auto tens_buffer = input_buffers.at(0);

        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

        for (const auto& core : cores) {
            {
                auto& runtime_args = writer_runtime_args[core.x][core.y];
                runtime_args[0] = tens_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
