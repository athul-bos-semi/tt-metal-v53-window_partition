// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include "tt_metal/common/core_coord.h"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include <sstream>
#include <type_traits>

#include "ttnn/cpp/ttnn/operations/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"


using namespace tt::constants;

namespace ttnn {

using namespace ccl;


std::tuple<std::vector<CoreCoord>, std::vector<uint32_t>> setup_datacopy(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& all_gather_output_tensor,
    Tensor& datacopy_output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    all_gather_op::Topology topology,

    CoreCoord datacopy_core_coord,
    uint32_t* all_gather_signal_semaphore_addr_ptr
) {

    auto const& all_gather_config = AllGatherConfig(input_tensor, all_gather_output_tensor, dim, ring_size, num_links, topology);
    const uint32_t num_transfers = 4; // ring_size - 1;

    auto tensor_slicer = ttnn::ccl::InterleavedRingAllGatherTensorSlicer (
        input_tensor,
        all_gather_output_tensor,
        dim,
        ring_index
    );


    // Setup cores used for datacopy
    std::vector<CoreCoord> all_datacopy_cores;
    all_datacopy_cores.reserve(1);


    // Select cores for datacopy (single core for now)
    CoreRangeSet datacopy_workers = CoreRangeSet({CoreRange(datacopy_core_coord)});

    auto datacopy_cores = corerange_to_cores(datacopy_workers, std::nullopt, true);
    all_datacopy_cores.insert(all_datacopy_cores.end(), datacopy_cores.begin(), datacopy_cores.end());

    std::cout << "Finished setting up datacopy cores." << std::endl;

    // Setup semaphores used to signal datacopy. TODO: instead of datacopy, this should be matmul cores
    auto datacopy_signal_semaphore_addr_dir0 = CreateSemaphore(program, datacopy_workers, 0);
    auto datacopy_signal_semaphore_addr_dir1 = CreateSemaphore(program, datacopy_workers, 0);

    // Setup args for the kernel
    const uint32_t tile_size = 32;
    const uint32_t page_size = all_gather_output_tensor.buffer()->page_size();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(all_gather_output_tensor.get_dtype());


    auto all_gather_output_buffer = all_gather_output_tensor.buffer();
    auto datacopy_output_buffer = datacopy_output_tensor.buffer();

    bool all_gather_output_is_dram = all_gather_output_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool datacopy_output_is_dram = datacopy_output_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t last_output_page_offset = (ring_size - 1) * tensor_slicer.output_page_offset;

    uint32_t num_rows = input_tensor.get_legacy_shape()[2] / tile_size ;

    // Core locations for the matmul [TODO]


    // Compile time args
    std::vector<uint32_t> datacopy_ct_args = {
        static_cast<uint32_t>(all_gather_output_is_dram),
        static_cast<uint32_t>(datacopy_output_is_dram),
        static_cast<uint32_t>(num_transfers),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(ring_index),
        static_cast<uint32_t>(ring_size),
        static_cast<uint32_t>(all_gather_output_tensor.get_legacy_shape()[3] / tile_size), // tesnor width
        static_cast<uint32_t>(all_gather_output_tensor.get_legacy_shape()[2] / tile_size), // tensor height
        static_cast<uint32_t>(tensor_slicer.num_cols), // tensor slice width in tiles
        static_cast<uint32_t>(num_rows), // tnesor slice height in tiles
        static_cast<uint32_t>(tensor_slicer.output_page_offset),
        static_cast<uint32_t>(last_output_page_offset),
        static_cast<uint32_t>(true ? 1 : 0), // TODO
        static_cast<uint32_t>(datacopy_signal_semaphore_addr_dir0),
        static_cast<uint32_t>(datacopy_signal_semaphore_addr_dir1),
        static_cast<uint32_t>(*all_gather_signal_semaphore_addr_ptr),
    };

    uint32_t cb_id_in0 = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_in0_config =
        tt::tt_metal::CircularBufferConfig(
            page_size * 200 /* TODO: Update to be actual number */, {{cb_id_in0, cb_data_format}})
            .set_page_size(cb_id_in0, page_size);
    auto cb_input = tt::tt_metal::CreateCircularBuffer(program, datacopy_workers, cb_in0_config);

    // Runtime args
    std::vector<uint32_t> datacopy_rt_args = {
        static_cast<uint32_t>(all_gather_output_buffer->address()),
        static_cast<uint32_t>(datacopy_output_buffer->address()),
    };

    std::map<string, string> kernel_defines = {
        {"TILED_LAYOUT", "1"},
        {"INTERLEAVED_MEM_LAYOUT", "1"}
    };


    // Create the kernel
    tt::tt_metal::KernelHandle datacopy_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather_matmul/device/kernels/datacopy.cpp",
        datacopy_workers,
        tt::tt_metal::WriterDataMovementConfig(datacopy_ct_args, kernel_defines));

    // Set runtime args
    tt::tt_metal::SetRuntimeArgs(
        program,
        datacopy_kernel_id,
        datacopy_workers,
        datacopy_rt_args
    );

    // Return the core coordinates and semaphore address
    return {all_datacopy_cores, {datacopy_signal_semaphore_addr_dir0, datacopy_signal_semaphore_addr_dir1}};
}


// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
operation::ProgramWithCallbacks all_gather_matmul_multi_core_with_workers(
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    Tensor& datacopy_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,

    /* All Gather Params */
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    all_gather_op::Topology topology,
    const CoreCoord core_grid_offset,


    /* Matmul Params */
    const std::optional<const Tensor> bias,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out

) {

    tt::tt_metal::Program program{};
    uint32_t* all_gather_signal_semaphore_addr_ptr = nullptr;

    auto matmul_program_with_callbacks = matmul_multi_core_reuse_mcast_2d_optimized_helper(
        program,
        datacopy_output_tensor,
        weight_tensor,
        bias,
        matmul_output_tensor,
        bcast_batch,
        compute_with_storage_grid_size,
        compute_kernel_config,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        fuse_batch,
        transpose_mcast,
        fused_activation,
        untilize_out,
        all_gather_signal_semaphore_addr_ptr);

    auto datacopy_params = setup_datacopy(matmul_program_with_callbacks.program,
        input_tensor,
        all_gather_output_tensor,
        datacopy_output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        {0, 5},
        all_gather_signal_semaphore_addr_ptr);

    const std::vector<CoreCoord>& datacopy_cores = std::get<0>(datacopy_params);
    const std::vector<uint32_t> datacopy_signal_semaphore_addr = std::get<1>(datacopy_params);

    // Pass in the datacopy cores and sempahore address (Using optional arguments)
    auto all_gather_program_with_callbacks = all_gather_multi_core_with_workers_helper(
        matmul_program_with_callbacks.program,
        input_tensor,
        all_gather_output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        receiver_device_id,
        sender_device_id,
        topology,
        datacopy_cores,
        datacopy_signal_semaphore_addr,
        core_grid_offset);




    // TODO: combine the program with callbacks from all-gather and matmul and return it as one single ProgramWithCallbacks

    return all_gather_program_with_callbacks;
}

}  // namespace ttnn
