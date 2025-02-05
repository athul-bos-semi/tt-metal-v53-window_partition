// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

#include "tt_metal/hw/inc/ethernet/dataflow_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_edm_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header_validate.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_types.hpp"
#include "debug/assert.h"
#include "debug/dprint.h"
#include <cstdint>

namespace tt::fabric {

/*
 * The WorkerToFabricEdmSender acts as an adapter between the worker and the EDM, it hides details
 * of the communication between worker and EDM to provide flexibility for the implementation to change
 * over time without kernel updates. Additionally, details for adapter setup w.r.t runtime args is also hidden.
 * The main functionality provided is:
 * - Opening a connection with the EDM
 * - Closing a connection with the EDM
 * - Flow control protocol between worker and EDM
 *
 * ### Flow Control Protocol:
 * The flow control protocol is rd/wr ptr based and is implemented as follows (from the worker's perspective):
 * The adapter has a local write pointer (wrptr) which is used to track the next buffer slot to write to. The adapter
 * also has a local memory slot that holds the remote read pointer (rdptr) of the EDM. The adapter uses the difference
 * between these two pointers (where rdptr trails wrptr) to determine if the EDM has space to accept a new packet.
 *
 * As the adapter writes into the EDM, it updates the local wrptr. As the EDM reads from its local L1 channel buffer,
 * it will notify the worker/adapter (here) by updating the worker remote_rdptr to carry the value of the EDM rdptr.
 */
struct WorkerToFabricEdmSender {
    static constexpr uint32_t unused_connection_value = 0;
    static constexpr uint32_t open_connection_value = 1;
    static constexpr uint32_t close_connection_request_value = 2;

    WorkerToFabricEdmSender() : from_remote_buffer_slot_rdptr_ptr(nullptr) {}

    template <ProgrammableCoreType my_core_type>
    static WorkerToFabricEdmSender build_from_args(std::size_t& arg_idx) {
        bool is_persistent_fabric = get_arg_val<uint32_t>(arg_idx++);
        WorkerXY const edm_worker_xy = WorkerXY::from_uint32(get_arg_val<uint32_t>(arg_idx++));
        auto const edm_buffer_base_addr = get_arg_val<uint32_t>(arg_idx++);
        uint8_t const num_buffers_per_channel = get_arg_val<uint32_t>(arg_idx++);
        size_t const edm_l1_sem_id = get_arg_val<uint32_t>(arg_idx++);
        auto const edm_connection_handshake_l1_addr = get_arg_val<uint32_t>(arg_idx++);
        auto const edm_worker_location_info_addr = get_arg_val<uint32_t>(arg_idx++);
        uint16_t const buffer_size_bytes = get_arg_val<uint32_t>(arg_idx++);
        auto const edm_buffer_index_addr = get_arg_val<uint32_t>(arg_idx++);
        auto writer_send_sem_addr =
            reinterpret_cast<volatile uint32_t* const>(get_semaphore<my_core_type>(get_arg_val<uint32_t>(arg_idx++)));
        auto worker_teardown_sem_addr =
            reinterpret_cast<volatile uint32_t* const>(get_semaphore<my_core_type>(get_arg_val<uint32_t>(arg_idx++)));
        auto const worker_buffer_index_semaphore_addr = get_semaphore<my_core_type>(get_arg_val<uint32_t>(arg_idx++));
        ASSERT(
            (my_core_type == ProgrammableCoreType::TENSIX && worker_buffer_index_semaphore_addr < 1499136) ||
            (my_core_type == ProgrammableCoreType::ACTIVE_ETH && worker_buffer_index_semaphore_addr < 262144));
        ASSERT(
            (my_core_type == ProgrammableCoreType::TENSIX && (uint32_t)writer_send_sem_addr < 1499136) ||
            (my_core_type == ProgrammableCoreType::ACTIVE_ETH && (uint32_t)writer_send_sem_addr < 262144));
        ASSERT(edm_buffer_index_addr < 262144);
        return WorkerToFabricEdmSender(
            is_persistent_fabric,
            edm_worker_xy.x,
            edm_worker_xy.y,
            edm_buffer_base_addr,
            num_buffers_per_channel,
            edm_l1_sem_id,
            edm_connection_handshake_l1_addr,
            edm_worker_location_info_addr,  // The EDM's location for `EDMChannelWorkerLocationInfo`
            buffer_size_bytes,
            edm_buffer_index_addr,
            writer_send_sem_addr,
            worker_teardown_sem_addr,
            worker_buffer_index_semaphore_addr);
    }

    WorkerToFabricEdmSender(
        bool connected_to_persistent_fabric,
        uint8_t edm_worker_x,
        uint8_t edm_worker_y,
        std::size_t edm_buffer_base_addr,
        uint8_t num_buffers_per_channel,
        size_t edm_l1_sem_id,  // may also be an address
        std::size_t edm_connection_handshake_l1_id,
        std::size_t edm_worker_location_info_addr,  // The EDM's location for `EDMChannelWorkerLocationInfo`
        uint16_t buffer_size_bytes,
        size_t edm_buffer_index_id,
        volatile uint32_t* const from_remote_buffer_slot_rdptr_ptr,
        volatile uint32_t* const worker_teardown_addr,
        uint32_t local_buffer_index_addr) :
        edm_buffer_addr(edm_buffer_base_addr),
        edm_buffer_slot_wrptr_addr(
            connected_to_persistent_fabric ? edm_l1_sem_id
                                           : get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(edm_l1_sem_id)),
        edm_connection_handshake_l1_addr(
            connected_to_persistent_fabric
                ? edm_connection_handshake_l1_id
                : get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(edm_connection_handshake_l1_id)),
        edm_worker_location_info_addr(edm_worker_location_info_addr),
        edm_buffer_index_addr(
            connected_to_persistent_fabric ? edm_buffer_index_id
                                           : get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(edm_buffer_index_id)),
        from_remote_buffer_slot_rdptr_ptr(from_remote_buffer_slot_rdptr_ptr),
        worker_teardown_addr(worker_teardown_addr),
        edm_buffer_base_addr(edm_buffer_base_addr),
        buffer_slot_wrptr_ptr(reinterpret_cast<size_t*>(local_buffer_index_addr)),
        buffer_size_bytes(buffer_size_bytes),
        num_buffers_per_channel(num_buffers_per_channel),
        last_buffer_index(num_buffers_per_channel - 1),
        edm_noc_x(edm_worker_x),
        edm_noc_y(edm_worker_y) {
        ASSERT(buffer_size_bytes > 0);
    }

    FORCE_INLINE bool edm_has_space_for_packet() const {
        auto const wrptr = *this->buffer_slot_wrptr_ptr;
        auto const rdptr = *this->from_remote_buffer_slot_rdptr_ptr;
        bool wrptr_ge_rptr = wrptr >= rdptr;
        uint8_t slots_used = wrptr_ge_rptr ? (wrptr - rdptr) : ((2 * this->num_buffers_per_channel) - rdptr) + wrptr;
        return slots_used < this->num_buffers_per_channel;
    }

    FORCE_INLINE void wait_for_empty_write_slot() const {
        while (!this->edm_has_space_for_packet());
    }

    FORCE_INLINE void send_payload_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        send_payload_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::BLOCKING>(cb_id, num_pages, page_size);
    }

    // Does not wait for CB. Assumes caller handles CB data availability
    FORCE_INLINE void send_payload_non_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        send_payload_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::NON_BLOCKING>(cb_id, num_pages, page_size);
    }

    /*
     * No CB
     */
    FORCE_INLINE void send_packet_header_and_notify_fabric_flush_blocking(uint32_t source_address) {
        send_packet_header_and_notify_fabric<ttnn::ccl::EDM_IO_BLOCKING_MODE::FLUSH_BLOCKING>(source_address);
    }
    FORCE_INLINE void send_payload_without_header_non_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_without_header_from_address_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::NON_BLOCKING>(source_address, size_bytes);
    }
    FORCE_INLINE void send_payload_flush_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_from_address_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::FLUSH_BLOCKING>(source_address, size_bytes);
    }
    FORCE_INLINE void send_payload_flush_non_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_from_address_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::NON_BLOCKING>(source_address, size_bytes);
    }
    FORCE_INLINE void send_payload_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_from_address_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::BLOCKING>(source_address, size_bytes);
    }

    /*
     * No CB
     */
    // Does not wait for CB. Assumes caller handles CB data availability
    FORCE_INLINE void send_payload_non_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_from_address_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::NON_BLOCKING>(source_address, size_bytes);
    }

    static constexpr size_t edm_sender_channel_field_stride_bytes = 16;

    void open() {
        const auto dest_noc_addr_coord_only = get_noc_addr(this->edm_noc_x, this->edm_noc_y, 0);

        const uint64_t remote_buffer_index_addr = dest_noc_addr_coord_only | edm_buffer_index_addr;
        ASSERT(remote_buffer_index_addr > 0);
        noc_async_read(remote_buffer_index_addr, reinterpret_cast<size_t>(this->buffer_slot_wrptr_ptr), sizeof(uint32_t));

        tt::fabric::EDMChannelWorkerLocationInfo* worker_location_info_ptr = reinterpret_cast<tt::fabric::EDMChannelWorkerLocationInfo*>(edm_worker_location_info_addr);
        const uint64_t edm_rdptr_addr = dest_noc_addr_coord_only | reinterpret_cast<size_t>(edm_worker_location_info_addr + offsetof(tt::fabric::EDMChannelWorkerLocationInfo, edm_rdptr));
        noc_async_read(edm_rdptr_addr, reinterpret_cast<size_t>(this->from_remote_buffer_slot_rdptr_ptr), sizeof(uint32_t));
        // TODO: Need to change byte enable to be word enable
        const uint64_t dest_edm_location_info_addr            = dest_noc_addr_coord_only | edm_worker_location_info_addr;
        const uint64_t edm_teardown_semaphore_address_address = dest_noc_addr_coord_only | reinterpret_cast<uint64_t>(&(worker_location_info_ptr->worker_teardown_semaphore_address));
        const uint64_t connection_worker_xy_address           = dest_noc_addr_coord_only | reinterpret_cast<uint64_t>(&(worker_location_info_ptr->worker_xy));
        noc_inline_dw_write(dest_edm_location_info_addr,            reinterpret_cast<size_t>(from_remote_buffer_slot_rdptr_ptr));
        noc_inline_dw_write(edm_teardown_semaphore_address_address, reinterpret_cast<size_t>(worker_teardown_addr));
        noc_inline_dw_write(connection_worker_xy_address,           ttnn::ccl::WorkerXY(my_x[0], my_y[0]).to_uint32());

        const uint64_t edm_connection_handshake_noc_addr = dest_noc_addr_coord_only | edm_connection_handshake_l1_addr;
        noc_inline_dw_write(edm_connection_handshake_noc_addr, open_connection_value);
        noc_async_read_barrier();
        ASSERT(*this->buffer_slot_wrptr_ptr < 20);
    }

    void close() {
        const auto dest_noc_addr_coord_only =
            get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_slot_wrptr_addr) & ~(uint64_t)NOC_COORDINATE_MASK;

        const uint64_t dest_edm_connection_state_addr = dest_noc_addr_coord_only | edm_connection_handshake_l1_addr;
        noc_inline_dw_write(dest_edm_connection_state_addr, close_connection_request_value);

        // buffer index stored at location after handshake addr
        const uint64_t remote_buffer_index_addr = dest_noc_addr_coord_only | edm_buffer_index_addr;
        noc_inline_dw_write(remote_buffer_index_addr, *this->buffer_slot_wrptr_ptr);

        // Need to wait for the ack to teardown notice, from edm
        noc_semaphore_wait(this->worker_teardown_addr, 1);

        noc_async_write_barrier();
    }

    uint32_t edm_buffer_addr;

    // the L1 address of buffer_slot wrptr on the EDM we are writing to
    // Writing to this address will tell the EDM that the wrptr is changed and
    // that new data is available
    uint32_t edm_buffer_slot_wrptr_addr;
    size_t edm_connection_handshake_l1_addr;
    size_t edm_worker_location_info_addr;
    size_t edm_buffer_index_addr;

    // Local copy of the the buffer slot rdptr on the EDM
    // EDM will update this to indicate that packets have been read (and hence
    // space is available)
    volatile uint32_t* from_remote_buffer_slot_rdptr_ptr;
    volatile uint32_t* worker_teardown_addr;
    size_t edm_buffer_base_addr;

    // TODO: keep a local copy that we use during the lifetime of the channel to avoid repeated L1 reads
    size_t* buffer_slot_wrptr_ptr;

    uint16_t buffer_size_bytes;
    uint8_t num_buffers_per_channel;

    // Specifies how many buffer slots are available in the EDM channel
    uint8_t last_buffer_index;

    // noc location of the edm we are connected to (where packets are sent to)
    uint8_t edm_noc_x;
    uint8_t edm_noc_y;

private:

    FORCE_INLINE void update_edm_buffer_slot_wrptr() {
        uint64_t const noc_sem_addr = get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_slot_wrptr_addr);

        noc_inline_dw_write(noc_sem_addr, *this->buffer_slot_wrptr_ptr);
    }

    FORCE_INLINE void advance_buffer_slot_wrptr() {
        // TODO: smarter addition if we are working with pow2
        uint8_t wrptr = *this->buffer_slot_wrptr_ptr;
        *this->buffer_slot_wrptr_ptr =
            !(wrptr == ((this->num_buffers_per_channel * 2) - 1)) ? wrptr + 1 : 0;
    }

    FORCE_INLINE uint8_t get_buffer_slot_index() const {
        auto const wrptr = *this->buffer_slot_wrptr_ptr;
        bool normalize = wrptr >= this->num_buffers_per_channel;
        return wrptr - (normalize * this->num_buffers_per_channel);
    }

    template <ttnn::ccl::EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_packet_header_and_notify_fabric(uint32_t source_address) {

        uint64_t buffer_address = get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_addr) +
                                  (this->get_buffer_slot_index() * (this->buffer_size_bytes + sizeof(eth_channel_sync_t)));

        send_chunk_from_address<blocking_mode>(source_address, 1, sizeof(tt::fabric::PacketHeader), buffer_address);
        this->advance_buffer_slot_wrptr();
        this->update_edm_buffer_slot_wrptr();
    }
    template <ttnn::ccl::EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_payload_without_header_from_address_impl(uint32_t source_address, size_t size_bytes) {
        uint64_t buffer_address = get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_addr) +
                                  (this->get_buffer_slot_index() * (this->buffer_size_bytes + sizeof(eth_channel_sync_t)));

        // skip past the first part of the buffer which will be occupied by the packet header
        send_chunk_from_address<blocking_mode>(source_address, 1, size_bytes, buffer_address + sizeof(tt::fabric::PacketHeader));
    }

    template <ttnn::ccl::EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_payload_from_address_impl(uint32_t source_address, size_t size_bytes) {
        uint64_t buffer_address = get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_addr) +
                                  (this->get_buffer_slot_index() * (this->buffer_size_bytes + sizeof(eth_channel_sync_t)));

        ASSERT(size_bytes <= this->buffer_size_bytes);
        ASSERT(tt::fabric::is_valid(*const_cast<tt::fabric::PacketHeader*>(
            reinterpret_cast<volatile tt::fabric::PacketHeader*>(source_address))));
        send_chunk_from_address<blocking_mode>(source_address, 1, size_bytes, buffer_address);

        this->advance_buffer_slot_wrptr();
        this->update_edm_buffer_slot_wrptr();
    }

    template <ttnn::ccl::EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_payload_impl(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        uint64_t buffer_address = get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_addr) +
                                  (this->get_buffer_slot_index() * (this->buffer_size_bytes + sizeof(eth_channel_sync_t)));
        ASSERT(num_pages * page_size <= this->buffer_size_bytes);
        send_chunk<blocking_mode>(cb_id, num_pages, page_size, buffer_address);
        this->advance_buffer_slot_wrptr();
        this->update_edm_buffer_slot_wrptr();
    }
};

}  // namespace tt::fabric
