// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace tt::tt_metal {
inline namespace v0 {

// Forward declarations
class Device;

}  // namespace v0
}  // namespace tt::tt_metal

namespace ttnn::ccl {
class WorkerEdmInterfaceArgs;
class SenderWorkerAdapterSpec;

namespace worker_detail {

Shape4D<uint32_t> to_4d_shape(Shape4D<uint32_t> const& shape);
Shape4D<uint32_t> to_4d_offset(Shape4D<uint32_t> const& offset);
size_t get_volume(Shape4D<uint32_t> const& shape);

Shape4D<uint32_t> to_4d_shape(tt_xy_pair const& shape);
Shape4D<uint32_t> to_4d_offset(tt_xy_pair const& offset);
size_t get_volume(tt_xy_pair const& shape);

void generate_ccl_slice_sequence_commands(
    std::vector<TensorSlice> const& slices,
    ttnn::ccl::cmd::CclCommandCode command_type,
    std::vector<uint32_t>& args_out);
void generate_ccl_slice_sequence_commands(
    std::vector<v2::TensorSlice> const& slices,
    ttnn::ccl::cmd::CclCommandCode command_type,
    std::vector<uint32_t>& args_out,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args);
void emit_ccl_send_slice_sequence_commands(std::vector<v1::TensorSlice> const& slices, std::vector<uint32_t>& args_out);
void generate_ccl_read_to_cb_slice_sequence_commands(
    std::vector<v2::TensorSlice> const& slices,
    std::vector<uint32_t>& args_out,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args);
void generate_ccl_cb_to_tensor_slice_sequence_commands(
    std::vector<v2::TensorSlice> const& slices,
    std::vector<uint32_t>& args_out,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args);
void generate_ccl_command_stream_to_kernel_args(
    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> const& ccl_command_stream,
    std::vector<uint32_t>& args_out);

// TODO: eventually take a fabric handle
void generate_multi_input_command_stream_kernel_rt_args(
    Program& program,
    KernelHandle kernel_id,
    std::vector<Tensor const*> const& tensors,
    std::vector<size_t> const& page_sizes,
    Device* device,
    uint32_t num_pages_per_edm_buffer,  // TODO: get from fabric
    CoreRangeSet const& worker_core_range,
    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> const& ccl_command_stream0,
    std::optional<std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand>> const& ccl_command_stream1,
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> const& forward_fabric_connections,
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> const& backward_fabric_connections,
    std::optional<std::unordered_map<const Tensor*, Device*>> const& tensor_device_override = std::nullopt);
// Helper functions for building command processing datamovement kernels
// TODO: Bundle into command bundle per command stream to cut down
//       on args and improve usability
void generate_multi_command_stream_kernel_rt_args(
    Program& program,
    KernelHandle kernel_id,
    std::vector<uint32_t> const& cb_ids,
    std::vector<const Tensor*> const& tensors,
    Device* device,
    uint32_t page_size,  // TODO: get from tensors
    CoreRangeSet const& worker_core_range,
    uint32_t num_pages_per_edm_buffer,  // TODO: get from fabric
    std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> const& command_tensor_slices,
    ttnn::ccl::cmd::CclCommandCode command_type,  // TODAY REQURED TO BE SAME - FUTURE - wrapped with above
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> const& forward_fabric_connections,
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> const& backward_fabric_connections,
    std::optional<std::vector<ttnn::ccl::edm_termination_info_t>> const& edm_termination_infos,
    std::vector<ttnn::ccl::cmd::CclCommandDestArgs> const& dest_args);
KernelHandle generate_multi_command_stream_kernel_ct_args(
    Program& program,
    std::vector<uint32_t> const& cb_indices,
    std::vector<Tensor const*> const& tensors,
    CoreRangeSet const& worker_core_range,
    DataMovementConfig datamovement_kernel_config,
    const size_t num_command_streams = 2,
    std::optional<chip_id_t> my_chip_id = std::nullopt);

// Maybe not the right place for this - re-evaluate
// Generates the kernel that allows async-tensor-mode CCLs to run in synchronous mode such that
// they will wait for all outstanding writes to complete before completing the CCL on any given chip
// to avoid races because, generally speaking, async mode for CCLs requires the consumer ops to support
// async tensors.
//

// Async tensor mode doesn't require that the producer of a tensor wait for the tensor to be fully populated
// before terminating; instead that responsibility is left to the consumer. This can be advantageous because it
// a) Allows dispatch overheads to be partly or fully hidden
// b) Allows producer and consumer ops to more natively overlap execution
void build_sync_kernels(
    Device* device,
    tt::tt_metal::Program& program,
    ccl::SyncModeSpec const& sync_details,
    bool terminate_fabric,
    ccl::EdmLineFabricOpInterface& fabric_interface);
ttnn::ccl::cmd::CclHostLowLevelCommandSequence build_ccl_cmd_proc_teardown_commands(
    tt::tt_metal::Program& program,
    Device* device,
    Device* forward_device,
    size_t line_size,
    size_t line_index,
    std::vector<ttnn::ccl::edm_termination_info_t> const& edm_termination_infos,
    ccl::SyncModeSpec const& sync_details,
    ccl::EdmLineFabricOpInterface& fabric_interface);

struct CCLWorkerArgBuilder {
    CCLWorkerArgBuilder(
        tt::tt_metal::Device const* device,
        ttnn::ccl::CCLOpConfig const& op_config,
        ttnn::ccl::TensorPartition const& input_tensor_partition,
        ttnn::ccl::TensorPartition const& output_tensor_partition,
        std::size_t operating_dim);

    std::vector<uint32_t> generate_sender_reader_kernel_rt_args(
        ttnn::ccl::InterleavedTensorWorkerSlice worker_slice,
        std::size_t operating_dim,
        uint32_t num_pages_per_packet,
        uint32_t worker_slice_index) const;

    std::vector<uint32_t> generate_sender_writer_kernel_rt_args(
        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> const& forward_fabric_connection,
        const size_t sender_worker_forward_flow_control_semaphore_id,
        const size_t sender_worker_forward_buffer_index_semaphore_id,
        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> const& backward_fabric_connection,
        const size_t sender_worker_backward_flow_control_semaphore_id,
        const size_t sender_worker_backward_buffer_index_semaphore_id,
        const size_t forward_direction_distance_to_end_of_line,
        const size_t backward_direction_distance_to_end_of_line,
        ttnn::ccl::InterleavedTensorWorkerSlice worker_slice,
        std::size_t operating_dim,
        uint32_t num_pages_per_packet,
        uint32_t worker_slice_index,
        std::optional<ttnn::ccl::SyncModeSpec> sync_details) const;

    std::vector<uint32_t> generate_sender_reader_kernel_ct_args() const;

    std::vector<uint32_t> generate_sender_writer_kernel_ct_args() const;

    tt::tt_metal::Device const* device;
    ttnn::ccl::TensorPartition const input_tensor_partition;
    ttnn::ccl::TensorPartition const output_tensor_partition;
    ttnn::ccl::CCLOpConfig const op_config;
    std::size_t operating_dim;
    bool src_is_dram;
    bool dst_is_dram;
};

}  // namespace worker_detail
}  // namespace ttnn::ccl
