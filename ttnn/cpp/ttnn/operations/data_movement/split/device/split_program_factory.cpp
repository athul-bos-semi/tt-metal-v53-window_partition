// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iterator>
#include <ranges>
#include <vector>
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/cpp/ttnn/operation.hpp"
namespace   ttnn::operations::data_movement::detail {

// operation for generic n-way splits where n != 2
operation::ProgramWithCallbacks split_last_dim_n_chunks_tiled(
    const Tensor &input_tensor, std::vector<Tensor> &output_tensors, const MemoryConfig &mem_config) {
    uint32_t dim = 3; // this op always splits on dim 3 for now.
    uint32_t num_chunks = output_tensors.size();

    auto input_shape = input_tensor.get_legacy_shape();

    Program program{};
    tt::tt_metal::Device *device = input_tensor.device();
    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(input_data_format);
    tt::tt_metal::Buffer *in0_buffer = input_tensor.buffer();

    // Output buffers
    TT_FATAL(std::all_of(output_tensors.begin(),
                        output_tensors.end(),
                        [](tt::tt_metal::Tensor &t) {return t.buffer() != nullptr;}),
            "Output tensor buffers should be allocated on device!"); // FIXME?: could make this more specific

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t z = input_shape[1]; // channels
    uint32_t num_tiles_y_dim = input_shape[2] / tt::constants::TILE_HEIGHT; // how many tiles along height axis of the input tensor
    uint32_t num_tiles_x_dim = input_shape[3] / tt::constants::TILE_WIDTH; // how many tiles along width axis of the input tensor
    uint32_t num_cores_x_limit = device->compute_with_storage_grid_size().x;
    uint32_t num_cores_y_limit = device->compute_with_storage_grid_size().y;

    // We are splitting along the width (last dim/dim 3) of the tensor so we
    // parallelize along height (dim 2) and depth (dim 1).

    uint32_t num_cores = (num_cores_x_limit * num_cores_y_limit);

    auto [num_cores_used, y_tiles_per_core] = get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_y_dim, num_cores);

    uint32_t x_tiles_per_core = num_tiles_x_dim;
    uint32_t xy_tiles_per_core = y_tiles_per_core * x_tiles_per_core;
    uint32_t tiles_per_core = xy_tiles_per_core * z;

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t num_cores_c = std::gcd(num_cores_used, num_cores_x_limit);
    uint32_t num_cores_r = num_cores_used / num_cores_c;

    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_r - 1, (std::size_t)start_core_y + num_cores_c - 1}
    );

    bool tile_dtype_is_bfloat16 = input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool in0_is_dram = in0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    std::vector<tt::tt_metal::Buffer*> output_buffers(output_tensors.size());
    std::transform(output_tensors.begin(),
                   output_tensors.end(),
                   output_buffers.begin(),
                   [](Tensor &t) { return t.buffer(); });

    TT_FATAL(std::all_of(output_tensors.begin(),
                         output_tensors.end(),
                         [&](tt::tt_metal::Tensor &t)
                           {return t.buffer()->buffer_type() == output_tensors[0].buffer()->buffer_type();}),
             "Output buffers should be the same type");

    bool out_is_dram = output_buffers[0]->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t x_tiles_per_bank = num_tiles_x_dim / num_chunks;

    uint32_t z_stride_read = num_tiles_x_dim * num_tiles_y_dim; // increase z by going through an x-y plane.
    uint32_t y_stride_read = num_tiles_x_dim; // advance by the width to go down in height

    std::vector<uint32_t> reader_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)tile_dtype_is_bfloat16,
                                                      // by default in dram
                                                      (std::uint32_t)in0_is_dram,

                                                      // READER COMPILE TIME ARGS
                                                      (std::uint32_t)z,
                                                      (std::uint32_t)z_stride_read,
                                                      (std::uint32_t)y_stride_read,
                                                      (std::uint32_t)y_tiles_per_core,
                                                      (std::uint32_t)x_tiles_per_core,
                                                      (std::uint32_t)x_tiles_per_bank,
                                                      (std::uint32_t)num_chunks
                                                    };

    uint32_t z_stride_write = (num_tiles_x_dim * num_tiles_y_dim) / num_chunks;
    uint32_t y_stride_write = num_tiles_x_dim / num_chunks;

    std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)tile_dtype_is_bfloat16,
                                                      (std::uint32_t)out_is_dram,
                                                      (std::uint32_t)y_tiles_per_core,
                                                      (std::uint32_t)x_tiles_per_bank,
                                                      (std::uint32_t)z,
                                                      (std::uint32_t)z_stride_write,
                                                      (std::uint32_t)y_stride_write,
                                                      (std::uint32_t)num_chunks};

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/reader_tm_tile_layout_split_n_chunks.cpp",
        all_cores,
	tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/writer_tm_tile_layout_split_n_chunks.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, input_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    auto setup_reader_runtime_args =
        [xy_tiles_per_core,
         num_cores_r,
         num_chunks,
         &program,
         reader_kernel_id
         ](const CoreCoord &core, Buffer* in0_buffer) {
            uint32_t reader_core_id = core.y * num_cores_r + core.x;
            uint32_t start_tile = xy_tiles_per_core * reader_core_id; // start tile within the input tensor
            std::vector<uint32_t> reader_runtime_args = {
                (std::uint32_t)start_tile,
                (std::uint32_t)(in0_buffer->address()),  // in0_tensor_addr
                (std::uint32_t)reader_core_id,
            };

            tt::tt_metal::SetRuntimeArgs(program,
                                         reader_kernel_id,
                                         core,
                                         reader_runtime_args);
        };

    auto setup_writer_runtime_args =
        [xy_tiles_per_core,
         num_cores_r,
         num_chunks,
         &program,
         writer_kernel_id](const CoreCoord &core, const std::vector<Buffer*> &output_buffers) {
            std::vector<uint32_t> output_addrs;
            output_addrs.reserve(output_buffers.size());
            std::transform(output_buffers.begin(),
                           output_buffers.end(),
                           std::back_inserter(output_addrs),
                           [](tt::tt_metal::Buffer *buf) { return buf->address(); });

            uint32_t writer_core_id = core.y * num_cores_r + core.x;
            uint32_t writer_start_tile = (xy_tiles_per_core / num_chunks) * writer_core_id; // start tile within each output bank

            std::vector<uint32_t> control_args = {writer_core_id, writer_start_tile};
            std::vector<uint32_t> writer_runtime_args;
            writer_runtime_args.reserve(output_addrs.size()+control_args.size());

            writer_runtime_args.insert(writer_runtime_args.end(),
                                       output_addrs.begin(),
                                       output_addrs.end());

            // add writer_core_id and start_tile to the end of the writer_runtime_args
            writer_runtime_args.insert(writer_runtime_args.end(),
                                       control_args.begin(),
                                       control_args.end());

            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        };

    auto setup_runtime_args =
        [setup_reader_runtime_args,
         setup_writer_runtime_args]
        (const CoreCoord &core,
         Buffer* in0_buffer,
         const std::vector<Buffer*> &output_buffers) {
            setup_reader_runtime_args(core, in0_buffer);
            setup_writer_runtime_args(core, output_buffers);
        };

    for (auto core : all_cores) {
        setup_reader_runtime_args(core, in0_buffer);
        setup_writer_runtime_args(core, output_buffers);
    };

    auto override_runtime_args_callback =
        [all_cores,
         setup_runtime_args](const Program &program,
                             const std::vector<Buffer *> &input_buffers,
                             const std::vector<Buffer *> &output_buffers) {
                                 const auto in0_buffer = input_buffers[0];
                                 for (auto core : all_cores) {
                                     setup_runtime_args(core, in0_buffer, output_buffers);
                                 }
                                 return;
                             };

    return {std::move(program), override_runtime_args_callback};
}

// operation for the special case of two-way splits
void setup_runtime(
    const Program &program,
    const uint32_t &core_offset,
    const uint32_t &num_cores_r,
    const uint32_t &num_cores_c,
    const uint32_t &z,
    const uint32_t &num_cores_x,
    const uint32_t &per_core_tiles_y,
    const uint32_t &per_core_tiles_x,
    const uint32_t &num_tiles_per_z,
    tt::tt_metal::Buffer *in0_buffer,
    tt::tt_metal::Buffer *out0_buffer,
    tt::tt_metal::Buffer *out1_buffer,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id) {
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    if (num_cores_c > 1) {
        TT_FATAL(num_cores_c % 2 == 0, "Must be even number of cores");
    }
    uint32_t idc_outer_limit = 1;
    uint32_t idc_inner_limit = num_cores_c;

    for (int id_r_outer = 0; id_r_outer < z; id_r_outer++) {
        for (int id_r_inner = 0; id_r_inner < num_cores_x; id_r_inner++) {
            uint32_t id_r = id_r_outer * num_cores_x + id_r_inner;

            uint32_t id_r_reader =
                id_r_outer * num_tiles_per_z + id_r_inner * per_core_tiles_y * num_cores_c * per_core_tiles_x;
            uint32_t id_r_writer = id_r_reader / 2;
            if (num_cores_c > 1) {
                idc_outer_limit = 2;
                idc_inner_limit = num_cores_c / 2;
            }
            for (int id_c_outer = 0; id_c_outer < idc_outer_limit; id_c_outer++) {
                for (int id_c_inner = 0; id_c_inner < idc_inner_limit; id_c_inner++) {
                    uint32_t id_c = id_c_outer * idc_inner_limit + id_c_inner;
                    CoreCoord core = {(std::size_t)start_core_x + id_r, (std::size_t)start_core_y + id_c};

                    uint32_t reader_core_id = id_c * per_core_tiles_y;
                    reader_core_id += id_r_reader;

                    std::vector<uint32_t> reader_runtime_args = {
                        (std::uint32_t)reader_core_id,
                        (std::uint32_t)(in0_buffer->address()),  // in0_tensor_addr
                        (std::uint32_t)0                         // split on last dim
                    };
                    bool out0_only = false;
                    bool out1_only = false;
                    if (num_cores_c > 1) {
                        out0_only = (id_c_outer == 0);
                        out1_only = (id_c_outer == 1);
                    }

                    uint32_t writer_core_id = id_c_inner * per_core_tiles_y + (id_r_writer);

                    std::vector<uint32_t> writer_runtime_args = {
                        writer_core_id,
                        (std::uint32_t)out0_buffer->address(),  // first base addr
                        (std::uint32_t)out1_buffer->address(),  // second base addr
                        (std::uint32_t)out0_only,
                        (std::uint32_t)out1_only};
                    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
                    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
                }
            }
        }
    }
}

operation::ProgramWithCallbacks split_last_dim_two_chunks_tiled(
    const Tensor &input_tensor, std::vector<Tensor> &output_tensors, const MemoryConfig &mem_config) {
    uint32_t dim = 3;
    uint32_t num_chunks = 2;

    auto input_shape = input_tensor.get_legacy_shape();

    Program program{};
    tt::tt_metal::Device *device = input_tensor.device();
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::tt_metal::Buffer *in0_buffer = input_tensor.buffer();

    // Output buffers
    TT_FATAL(output_tensors.size() == num_chunks);
    tt::tt_metal::Tensor &out0 = output_tensors[0];
    tt::tt_metal::Tensor &out1 = output_tensors[1];

    tt::tt_metal::Buffer *out0_buffer = out0.buffer();
    TT_FATAL(out0_buffer != nullptr, "Output 0 buffer should be allocated on device!");
    tt::tt_metal::Buffer *out1_buffer = out1.buffer();
    TT_FATAL(out1_buffer != nullptr, "Output 1 buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t z = input_shape[1];
    uint32_t num_tiles_dim_2 = input_shape[2] / tt::constants::TILE_HEIGHT;
    uint32_t num_tiles_dim_3 = input_shape[3] / tt::constants::TILE_WIDTH;
    uint32_t num_cores_x_limit = device->compute_with_storage_grid_size().x;
    uint32_t num_cores_y_limit = device->compute_with_storage_grid_size().y;

    // parallelize z
    auto num_cores_z = z;

    // parallelize y
    auto [num_cores_y, per_core_tiles_y] =
        get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_3, num_cores_y_limit, /*request_even=*/true);

    // parallelize x
    auto [num_cores_x, per_core_tiles_x] =
        get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_2, num_cores_x_limit / num_cores_z);

    uint32_t per_core_tiles = per_core_tiles_x * per_core_tiles_y * (z / num_cores_z);

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t num_cores_c = num_cores_y;
    uint32_t num_cores_r = num_cores_x * num_cores_z;

    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_r - 1, (std::size_t)start_core_y + num_cores_c - 1}
    );

    bool tile_dtype_is_bfloat16 = input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool in0_is_dram = in0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = out0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    TT_FATAL(out0_buffer->buffer_type() == out1_buffer->buffer_type(), "Output buffers should be the same type");

    uint32_t num_tiles_per_z = (per_core_tiles_x * num_cores_x) * (per_core_tiles_y * num_cores_y);
    uint32_t z_stride_read = num_tiles_per_z;
    uint32_t y_stride_read = per_core_tiles_y * num_cores_y;

    std::vector<uint32_t> reader_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)tile_dtype_is_bfloat16,
                                                      // by default in dram
                                                      (std::uint32_t)in0_is_dram,

                                                      // READER COMPILE TIME ARGS
                                                      (std::uint32_t)(z / num_cores_z),
                                                      (std::uint32_t)per_core_tiles_x,  // out_num_tiles_per_tensor
                                                      (std::uint32_t)per_core_tiles_y,  // out_num_tiles_per_tensor
                                                      (std::uint32_t)z_stride_read,
                                                      (std::uint32_t)y_stride_read};

    uint32_t z_stride_write = num_tiles_per_z / num_chunks;
    uint32_t y_stride_write = per_core_tiles_y * (num_cores_c / num_chunks);
    std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)tile_dtype_is_bfloat16,
                                                      (std::uint32_t)out_is_dram,

                                                      (std::uint32_t)per_core_tiles_x,  // out_num_tiles_per_tensor
                                                      (std::uint32_t)per_core_tiles_y,  // out_num_tiles_per_tensor

                                                      (std::uint32_t)(z / num_cores_z),
                                                      (std::uint32_t)z_stride_write,
                                                      (std::uint32_t)y_stride_write

    };

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/reader_tm_tile_layout_split_two_chunks.cpp",
        all_cores,
	tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/writer_tm_tile_layout_split_two_chunks.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    setup_runtime(
        program,
        0,
        num_cores_r,
        num_cores_c,
        num_cores_z,
        num_cores_x,
        per_core_tiles_y,
        per_core_tiles_x,
        num_tiles_per_z,
        in0_buffer,
        out0_buffer,
        out1_buffer,
        reader_kernel_id,
        writer_kernel_id);

    auto override_runtime_args_callback =
        [reader_kernel_id, writer_kernel_id, num_cores_r, num_cores_c, start_core_x, start_core_y](
            const Program &program,
            const std::vector<Buffer *> &input_buffers,
            const std::vector<Buffer *> &output_buffers) {
            auto src_dram_buffer = input_buffers.at(0);

            auto dst_0_dram_buffer = output_buffers.at(0);
            auto dst_1_dram_buffer = output_buffers.at(0);

            for (int core_idx_y = 0; core_idx_y < num_cores_c; core_idx_y++) {
                for (int core_idx_x = 0; core_idx_x < num_cores_r; core_idx_x++) {
                    CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

                    {
                        auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                        runtime_args[1] = src_dram_buffer->address();
                    }

                    {
                        auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                        runtime_args[1] = dst_0_dram_buffer->address();
                        runtime_args[2] = dst_1_dram_buffer->address();
                    }
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}


}
