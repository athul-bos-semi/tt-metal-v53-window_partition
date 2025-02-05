// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "dataflow_api.h"
#include "ethernet/dataflow_api.h"
#include "debug/assert.h"
#include "debug/dprint.h"

// #define ENABLE_DEBUG 1

struct eth_buffer_slot_sync_t {
    volatile uint32_t bytes_sent;
    volatile uint32_t receiver_ack;
    volatile uint32_t src_id;

    uint32_t reserved_2;
};

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

FORCE_INLINE void switch_context_if_debug() {
#if ENABLE_DEBUG
    internal_::risc_context_switch();
#endif
}

template <typename T>
bool is_power_of_two(T val) {
    return (val & (val - 1)) == T(0);
}

// ******************************* Common Ct Args ************************************************

constexpr uint32_t NUM_BUFFER_SLOTS = get_compile_time_arg_val(0);
constexpr uint32_t MAX_NUM_TRANSACTION_ID =
    NUM_BUFFER_SLOTS / 2;  // the algorithm only works for NUM_BUFFER_SLOTS divisible by MAX_NUM_TRANSACTION_ID
constexpr uint32_t worker_noc_x = get_compile_time_arg_val(1);
constexpr uint32_t worker_noc_y = get_compile_time_arg_val(2);
constexpr uint32_t worker_buffer_addr = get_compile_time_arg_val(3);

constexpr uint32_t SENDER_QNUM = 1;
constexpr uint32_t RECEIVER_QNUM = 1;

// ******************************* Sender APIs ***************************************************

FORCE_INLINE uint32_t setup_sender_buffer(
    std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t buffer_slot_addr,
    uint32_t message_size) {
    for (uint8_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        buffer_slot_addrs[i] = buffer_slot_addr;
        buffer_slot_addr += message_size;
        buffer_slot_sync_addrs[i] = reinterpret_cast<volatile eth_buffer_slot_sync_t*>(buffer_slot_addr);
        buffer_slot_addr += sizeof(eth_buffer_slot_sync_t);
    }

    // reset bytes_sent to 0s so first iter it won't block
    for (uint32_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        buffer_slot_sync_addrs[i]->bytes_sent = 0;
    }

    // assemble a packet filled with values
    for (uint32_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        tt_l1_ptr uint8_t* ptr = reinterpret_cast<tt_l1_ptr uint8_t*>(buffer_slot_addrs[i]);
        for (uint32_t j = 0; j < message_size; j++) {
            ptr[j] = j;
        }
    }

    uint32_t buffer_end_addr = buffer_slot_addr;
    return buffer_end_addr;
}

FORCE_INLINE uint32_t advance_buffer_slot_ptr(uint32_t curr_ptr) { return (curr_ptr + 1) % NUM_BUFFER_SLOTS; }

FORCE_INLINE void write_receiver(
    uint32_t buffer_slot_addr,
    volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr,
    uint32_t full_payload_size,
    uint32_t qnum) {
    buffer_slot_sync_addr->bytes_sent = 1;

    while (eth_txq_is_busy(qnum)) {
        switch_context_if_debug();
    }

    eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
        buffer_slot_addr, buffer_slot_addr, full_payload_size, qnum);
}

FORCE_INLINE bool has_receiver_ack(volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr) {
    return buffer_slot_sync_addr->bytes_sent == 0;
}

FORCE_INLINE void check_buffer_full_and_send_packet(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t read_ptr,
    uint32_t& write_ptr,
    uint64_t full_payload_size) {
    uint32_t next_write_ptr = advance_buffer_slot_ptr(write_ptr);
    bool buffer_not_full = next_write_ptr != read_ptr;

    if (buffer_not_full) {
        write_receiver(buffer_slot_addrs[write_ptr], buffer_slot_sync_addrs[write_ptr], full_payload_size, SENDER_QNUM);

        write_ptr = next_write_ptr;
    }
}

FORCE_INLINE void check_receiver_done(
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t& read_ptr,
    uint32_t& num_messages_ack) {
    if (has_receiver_ack(buffer_slot_sync_addrs[read_ptr])) {
        read_ptr = advance_buffer_slot_ptr(read_ptr);
        num_messages_ack++;
    }
}

FORCE_INLINE void update_sender_state(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t full_payload_size,
    uint32_t& num_messages_ack,
    uint32_t& buffer_read_ptr,
    uint32_t& buffer_write_ptr) {
    // Check if current buffer slot is ready and send packet to receiver
    check_buffer_full_and_send_packet(
        buffer_slot_addrs, buffer_slot_sync_addrs, buffer_read_ptr, buffer_write_ptr, full_payload_size);
    // Check if the write for trid is done, and ack sender if the current buffer slot is done
    check_receiver_done(buffer_slot_sync_addrs, buffer_read_ptr, num_messages_ack);
}

// ******************************* Receiver APIs *************************************************

FORCE_INLINE uint32_t setup_receiver_buffer(
    std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t buffer_slot_addr,
    uint32_t message_size) {
    for (uint8_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        buffer_slot_addrs[i] = buffer_slot_addr;
        buffer_slot_addr += message_size;
        buffer_slot_sync_addrs[i] = reinterpret_cast<volatile eth_buffer_slot_sync_t*>(buffer_slot_addr);
        buffer_slot_sync_addrs[i]->bytes_sent = 0;
        buffer_slot_sync_addrs[i]->receiver_ack = 0;
        buffer_slot_addr += sizeof(eth_buffer_slot_sync_t);
    }

    uint32_t buffer_end_addr = buffer_slot_addr;
    return buffer_end_addr;
}

FORCE_INLINE uint32_t get_buffer_slot_trid(uint32_t curr_ptr) { return curr_ptr % MAX_NUM_TRANSACTION_ID + 1; }

FORCE_INLINE bool has_incoming_packet(volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr) {
    return buffer_slot_sync_addr->bytes_sent != 0;
}

FORCE_INLINE bool write_worker_done(uint32_t trid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid);
}

FORCE_INLINE void ack_complete(volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr, uint32_t qnum) {
    buffer_slot_sync_addr->bytes_sent = 0;

    while (eth_txq_is_busy(qnum)) {
        switch_context_if_debug();
    }

    eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
        reinterpret_cast<uint32_t>(buffer_slot_sync_addr),
        reinterpret_cast<uint32_t>(buffer_slot_sync_addr),
        sizeof(eth_buffer_slot_sync_t),
        qnum);
}

FORCE_INLINE void write_worker(
    uint32_t buffer_slot_addr,
    volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr,
    uint64_t worker_noc_addr,
    uint32_t message_size,
    uint32_t curr_trid_to_write) {
    // write to local
    noc_async_write_one_packet_with_trid_with_state(
        buffer_slot_addr, worker_noc_addr, message_size, curr_trid_to_write);

    // reset sync
    buffer_slot_sync_addr->bytes_sent = 0;
}

FORCE_INLINE void check_incomping_packet_and_write_worker(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t read_ptr,
    uint32_t& write_ptr,
    uint64_t worker_noc_addr,
    uint32_t message_size) {
    uint32_t next_write_ptr = advance_buffer_slot_ptr(write_ptr);
    bool buffer_not_full = next_write_ptr != read_ptr;

    if (buffer_not_full && has_incoming_packet(buffer_slot_sync_addrs[write_ptr])) {
        uint32_t curr_trid = get_buffer_slot_trid(write_ptr);
        write_worker(
            buffer_slot_addrs[write_ptr], buffer_slot_sync_addrs[write_ptr], worker_noc_addr, message_size, curr_trid);

        write_ptr = next_write_ptr;
    }
}

FORCE_INLINE void check_write_worker_done_and_send_ack(
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t& read_ptr,
    uint32_t write_ptr,
    uint32_t& num_messages_ack) {
    bool buffer_not_empty = read_ptr != write_ptr;
    uint32_t curr_trid = get_buffer_slot_trid(read_ptr);

    if (buffer_not_empty && write_worker_done(curr_trid)) {
        ack_complete(buffer_slot_sync_addrs[read_ptr], RECEIVER_QNUM);

        read_ptr = advance_buffer_slot_ptr(read_ptr);

        num_messages_ack++;
    }
}

FORCE_INLINE void update_receiver_state(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint64_t worker_noc_addr,
    uint32_t message_size,
    uint32_t& num_messages_ack,
    uint32_t& buffer_read_ptr,
    uint32_t& buffer_write_ptr) {
    // Check if there's an incoming packet for current buffer slot and write to worker if there's new packet
    check_incomping_packet_and_write_worker(
        buffer_slot_addrs, buffer_slot_sync_addrs, buffer_read_ptr, buffer_write_ptr, worker_noc_addr, message_size);
    // Check if the write for trid is done, and ack sender if the current buffer slot is done
    check_write_worker_done_and_send_ack(buffer_slot_sync_addrs, buffer_read_ptr, buffer_write_ptr, num_messages_ack);
}
