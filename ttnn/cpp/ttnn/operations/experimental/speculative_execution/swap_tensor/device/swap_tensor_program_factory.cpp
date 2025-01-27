// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swap_tensor_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/core_coord.hpp>
// #include "eth_l1_address_map.h"
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include <sstream>
#include <type_traits>
#include <ranges>

#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"

#include <optional>
using ttnn::ccl::LineTopology;
using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::speculative_execution::detail {

std::tuple<KernelHandle, KernelHandle> ccl_multi_core_with_workers(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    const Tensor& output_tensor,
    uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    CoreCoord ccl_core,
    GlobalSemaphore& global_semaphore) {
    log_trace(tt::LogOp, "CCL idx: {}", ring_index);
    log_trace(tt::LogOp, "input_tensor addr: {}", input_tensor.buffer()->address());
    log_trace(tt::LogOp, "output_tensor addr: {}", output_tensor.buffer()->address());

    IDevice* device = input_tensor.device();
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> local_fabric_handle =
        ttnn::ccl::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
            device, forward_device, backward_device, &program, true /*enable_persistent_fabric_mode*/, num_links);
    LineTopology line_topology(ring_size, ring_index);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    log_trace(tt::LogOp, "op_config page size: {}", op_config.get_page_size());

    // Get worker cores, assuming 1 worker per link, 1 link
    auto ccl_core_range = CoreRangeSet(CoreRange(ccl_core, ccl_core));
    auto ccl_core_physical = device->worker_core_from_logical_core(ccl_core);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = local_fabric_handle->get_edm_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // tripple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    CBHandle cb_src0_workers = CreateCircularBuffer(program, ccl_core_range, cb_src0_config);

    // Create Tensor slicer
    // read the entire input tensor (partition size = 1, partition index = 0)
    // write to the output tensor on its corresponding partition (partition size = ring_size, partition index =
    // ring_index)
    auto input_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicerV2(
        input_tensor,
        3,         // not used. dim=3 (we are not slicing anything)
        0,         // partition index
        1,         // partition size
        num_links  // num_workers_per_slicer, set 1 for now
    );
    auto output_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicerV2(
        output_tensor,
        3,         // not used. dim=3 (we are not slicing anything)
        0,         // partition index
        1,         // partition size
        num_links  // num_workers_per_slicer, set 1 for now
    );

    // KERNEL CREATION
    KernelHandle worker_sender_reader_kernel_id =
        ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {src0_cb_index},
            {&input_tensor},
            ccl_core_range,
            tt::tt_metal::ReaderDataMovementConfig{},
            1,  // num_command_streams
            device->id());

    KernelHandle worker_sender_writer_kernel_id =
        ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {src0_cb_index},
            {&output_tensor},
            ccl_core_range,
            tt::tt_metal::WriterDataMovementConfig{},
            1,  // num_command_streams
            device->id());

    bool is_forward_direction = forward_device.has_value();
    TT_FATAL(
        is_forward_direction || backward_device.has_value(),
        "At least one of forward or backward device must be provided");
    ttnn::ccl::cmd::UnicastCommandDestArgs unicast_dest_args = {1, is_forward_direction};
    log_trace(
        tt::LogOp,
        "[unicast_dest_args] distance: {}, is_forward_direction: {}",
        unicast_dest_args.distance_in_hops,
        unicast_dest_args.is_forward_direction);

    const auto& input_worker_slice_v2 = input_tensor_slicer.get_worker_slice_v2(0);
    const auto& output_worker_slice_v2 = output_tensor_slicer.get_worker_slice_v2(0);

    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> forward_fabric_connection =
        line_topology.is_first_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
            ? std::nullopt
            : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(
                  local_fabric_handle->uniquely_connect_worker(device, ttnn::ccl::EdmLineFabricOpInterface::FORWARD));
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> backward_fabric_connection =
        line_topology.is_last_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
            ? std::nullopt
            : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(
                  local_fabric_handle->uniquely_connect_worker(device, ttnn::ccl::EdmLineFabricOpInterface::BACKWARD));

    log_trace(
        tt::LogOp,
        "DEBUG: line_index: {}, line_size: {}, forward_fabric_connection: {}",
        line_topology.line_index(),
        line_topology.line_size(),
        forward_fabric_connection.has_value());
    log_trace(
        tt::LogOp,
        "DEBUG: line_index: {}, line_size: {}, backward_fabric_connection: {}",
        line_topology.line_index(),
        line_topology.line_size(),
        backward_fabric_connection.has_value());

    // READER COMMAND STREAM and RT ARGS
    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> reader_cmd_stream;
    // 1, read the input tensor slice to the CB
    reader_cmd_stream.push_back(  // use the reader_tensor_slices after the bug is fixed
        ttnn::ccl::cmd::uops::read_tensor_slice_to_cb_for_eventual_fabric_write(input_worker_slice_v2, src0_cb_index));

    ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
        program,
        worker_sender_reader_kernel_id,
        {&input_tensor},
        {op_config.get_page_size()},
        input_tensor.device(),
        num_pages_per_packet,
        {ccl_core},
        reader_cmd_stream,
        std::nullopt,
        std::nullopt,
        std::nullopt);

    // WRITER COMMAND STREAM and RT ARGS
    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> writer_cmd_stream;
    // 1, do unicast of the tensor slice to its destination
    writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::fabric_write_cb_to_tensor_slice(
        output_worker_slice_v2, src0_cb_index, unicast_dest_args));
    // 2, unicast the semaphore to dest for ccl ready
    writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::fabric_unicast_semaphore_inc(
        &global_semaphore,
        ttnn::ccl::cmd::CclCommandAtomicInc{1},
        ccl_core_physical.x,
        ccl_core_physical.y,
        unicast_dest_args));
    // 3, wait for ccl result ready semaphore
    writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_semaphore_wait(&global_semaphore, 1));
    // 4, reset local global semaphore to 0
    writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_core_semaphore_set(&global_semaphore, 0));

    // set the rt args
    ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
        program,
        worker_sender_writer_kernel_id,
        {&output_tensor},
        {op_config.get_page_size()},
        output_tensor.device(),
        num_pages_per_packet,  // num_pages_per_edm_buffer
        {ccl_core},
        writer_cmd_stream,
        std::nullopt,
        {forward_fabric_connection},
        {backward_fabric_connection});

    return std::tuple{worker_sender_reader_kernel_id, worker_sender_writer_kernel_id};
}

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
operation::ProgramWithCallbacks swap_tensor(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    uint32_t num_links,
    uint32_t num_devices,
    uint32_t device_index,
    ttnn::ccl::Topology topology,
    GlobalSemaphore global_semaphore,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device) {
    Program program = CreateProgram();

    CoreCoord ccl_core = {0, 0};
    std::optional<KernelHandle> ccl_reader_kernel_id, ccl_writer_kernel_id;

    std::tie(ccl_reader_kernel_id, ccl_writer_kernel_id) = ccl_multi_core_with_workers(
        program,
        input_tensor,     // input tensor
        forward_device,   // forward device
        backward_device,  // backward device
        output_tensor,    // output tensor
        num_links,
        num_devices,        // ring size
        device_index,       // ring index
        topology,           // topology
        ccl_core,           // ccl core
        global_semaphore);  // global semaphore handle

    auto override_runtime_arguments_callback =
        [ccl_reader_kernel_id, ccl_writer_kernel_id, ccl_core](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto in_buffer = input_tensors.at(0).buffer();
            auto out_buffer = output_tensors.at(0).buffer();
            auto in_addr = in_buffer->address();
            auto out_addr = out_buffer->address();

            // Update ccl related runtime args
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, ccl_reader_kernel_id.value());
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, ccl_writer_kernel_id.value());
            // reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[ccl_core.x][ccl_core.y];
            worker_reader_sender_runtime_args.at(0) = in_addr;
            // writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[ccl_core.x][ccl_core.y];
            worker_writer_sender_runtime_args.at(0) = out_addr;
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::speculative_execution::detail
