// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"

constexpr ProgrammableCoreType fd_core_type = static_cast<ProgrammableCoreType>(FD_CORE_TYPE);

constexpr uint32_t PACKET_WORD_SIZE_BYTES = 16;
constexpr uint32_t NUM_WR_CMD_BUFS = 4;
constexpr uint32_t DEFAULT_MAX_NOC_SEND_WORDS = (NUM_WR_CMD_BUFS-1)*(NOC_MAX_BURST_WORDS*NOC_WORD_BYTES)/PACKET_WORD_SIZE_BYTES;
constexpr uint32_t DEFAULT_MAX_ETH_SEND_WORDS = 2*1024;
constexpr uint32_t FVC_SYNC_THRESHOLD = 256;

enum SessionCommand : uint32_t {
    ASYNC_WR = (0x1 << 0),
    ASYNC_WR_RESP = (0x1 << 1),
    ASYNC_RD = (0x1 << 2),
    ASYNC_RD_RESP = (0x1 << 3),
    DSOCKET_WR = (0x1 << 4),
    SSOCKET_WR = (0x1 << 5),
    ATOMIC_INC = (0x1 << 6),
    ATOMIC_READ_INC = (0x1 << 7),
};

#define INVALID 0x0
#define DATA 0x1
#define MCAST_DATA 0x2
#define SYNC 0x4
#define FORWARD 0x8
#define PACK_N_FORWARD 0x10
#define NOP 0xFF

typedef struct tt_routing {
    uint32_t    packet_size_bytes;
    uint16_t    dst_mesh_id; // Remote mesh
    uint16_t    dst_dev_id;  // Remote device
    uint16_t    src_mesh_id; // Source mesh
    uint16_t    src_dev_id;  // Source device
    uint16_t    ttl;
    uint8_t     version;
    uint8_t     flags;
} tt_routing;

static_assert(sizeof(tt_routing) == 16);

typedef struct tt_session {
    SessionCommand command;
    uint32_t target_offset_l; // RDMA address
    uint32_t target_offset_h;
    uint32_t ack_offset_l; // fabric client local address for session command acknowledgement.
                         // This is complete end-to-end acknowledgement of sessoin command completion at the remote device.
    uint32_t ack_offset_h;
} tt_session;

static_assert(sizeof(tt_session) == 20);

typedef struct mcast_params {
    uint16_t east;
    uint16_t west;
    uint16_t north;
    uint16_t south;
    uint32_t socket_id; // Socket Id for DSocket Multicast. Ignored for ASYNC multicast.
} mcast_params;

typedef struct socket_params {
    uint32_t socket_id;
} socket_params;

typedef struct atomic_params {
    uint32_t return_offset; // L1 offset where atomic read should be returned. Noc X/Y is taken from tt_session.ack_offset
    uint32_t wrap_boundary; // NOC atomic increment wrapping value.
} atomic_params;

typedef struct read_params {
    uint32_t return_offset_l; // address where read data should be copied
    uint32_t return_offset_h;
    uint32_t size; // number of bytes to read
} read_params;

typedef union packet_params {
    mcast_params mcast_parameters;
    socket_params socket_parameters;
    atomic_params atomic_parameters;
    read_params read_parameters;
    uint8_t bytes[12];
} packet_params;

typedef struct packet_header {
    packet_params packet_parameters;
    tt_session session;
    tt_routing routing;
} packet_header_t;

const uint32_t PACKET_HEADER_SIZE_BYTES = 48;
const uint32_t PACKET_HEADER_SIZE_WORDS = PACKET_HEADER_SIZE_BYTES / PACKET_WORD_SIZE_BYTES;

static_assert(sizeof(packet_header) == PACKET_HEADER_SIZE_BYTES);

// This is a pull request entry for a fabric router.
// Pull request issuer populates these entries to identify
// the data that fabric router needs to pull from requestor.
// This data is the forwarded by router over ethernet.
// A pull request can be for packetized data or raw data, as specified by flags field.
//   - When registering a pull request for raw data, the requestor pushes two entries to router request queue.
//     First entry is packet_header, second entry is pull_request. This is typical of OP/Endpoint issuing read/writes over tt-fabric.
//   - When registering a pull request for packetized data, the requetor only pushed pull_request entry to router request queue.
//     This is typical of fabric routers forwarding data over noc/ethernet hops.
//
typedef struct pull_request {
    uint32_t wr_ptr; // Current value of write pointer.
    uint32_t rd_ptr; // Current value of read pointer. Points to first byte of pull data.
    uint32_t size;  // Total number of bytes that need to be forwarded.
    uint32_t buffer_size; // Producer local buffer size. Used for flow control when total data to send does not fit in local buffer.
    uint64_t buffer_start; // Producer local buffer start. Used for wrapping rd/wr_ptr at the end of buffer.
    uint64_t ack_addr; // Producer local address to send rd_ptr updates. fabric router pushes its rd_ptr to requestor at this address.
    uint8_t  padding[15];
    uint8_t  flags; // Router command.
} pull_request_t;

const uint32_t PULL_REQ_SIZE_BYTES = 48;

static_assert(sizeof(pull_request) == PULL_REQ_SIZE_BYTES);
static_assert(sizeof(pull_request) == sizeof(packet_header));

typedef union chan_request_entry {
  pull_request_t pull_request;
  packet_header_t packet_header;
  uint8_t bytes[48];
} chan_request_entry_t;


const uint32_t CHAN_PTR_SIZE_BYTES = 16;
typedef struct chan_ptr{
  uint32_t ptr;
  uint32_t pad[3];
} chan_ptr;
static_assert(sizeof(chan_ptr) == CHAN_PTR_SIZE_BYTES);

const uint32_t CHAN_REQ_BUF_SIZE = 16; // must be 2^N
const uint32_t CHAN_REQ_BUF_SIZE_MASK = (CHAN_REQ_BUF_SIZE - 1);
const uint32_t CHAN_REQ_BUF_PTR_MASK  = ((CHAN_REQ_BUF_SIZE << 1) - 1);
const uint32_t CHAN_REQ_BUF_SIZE_BYTES = 2 * CHAN_PTR_SIZE_BYTES + CHAN_REQ_BUF_SIZE * PULL_REQ_SIZE_BYTES;

typedef struct chan_req_buf{
  chan_ptr wrptr;
  chan_ptr rdptr;
  chan_request_entry_t chan_req[CHAN_REQ_BUF_SIZE];
} chan_req_buf;

static_assert(sizeof(chan_req_buf) == CHAN_REQ_BUF_SIZE_BYTES);

typedef struct chan_payload_ptr{
  uint32_t ptr;
  uint32_t pad[2];
  uint32_t ptr_cleared;
} chan_payload_ptr;

static_assert(sizeof(chan_payload_ptr) == CHAN_PTR_SIZE_BYTES);

const uint32_t SYNC_BUF_SIZE = 16; // must be 2^N
const uint32_t SYNC_BUF_SIZE_MASK = (SYNC_BUF_SIZE - 1);
const uint32_t SYNC_BUF_PTR_MASK  = ((SYNC_BUF_SIZE << 1) - 1);

extern uint64_t xy_local_addr;
extern volatile pull_request_t pull_request;

typedef struct fvc_state {
    volatile chan_payload_ptr remote_rdptr;
    uint32_t remote_ptr_update_addr;
    uint8_t chan_num;
    uint8_t packet_in_progress;
    uint8_t sync_buf_wrptr;
    uint8_t sync_buf_rdptr;
    uint32_t packet_words_remaining;
    uint32_t fvc_out_wrptr;
    uint32_t fvc_out_rdptr;
    uint32_t fvc_pull_wrptr;
    uint32_t buffer_size;
    uint32_t buffer_start;
    uint32_t remote_buffer_start;
    uint32_t pull_words_in_flight;
    uint32_t words_since_last_sync;
    uint32_t words_to_forward;
    uint8_t sync_pending;
    uint8_t padding[3];
    uint32_t sync_buf[SYNC_BUF_SIZE];

    uint32_t get_num_words_free() {
        uint32_t rd_ptr = remote_rdptr.ptr;
        uint32_t words_occupied = 0;
        if (fvc_pull_wrptr != rd_ptr) {
            words_occupied = fvc_pull_wrptr > rd_ptr ? fvc_pull_wrptr - rd_ptr : buffer_size * 2 + fvc_pull_wrptr - rd_ptr;
        }
        return buffer_size - words_occupied;
    }

    uint32_t get_remote_num_words_free() {
        uint32_t rd_ptr = remote_rdptr.ptr_cleared;
        uint32_t words_occupied = 0;
        if (fvc_out_wrptr != rd_ptr) {
            words_occupied = fvc_out_wrptr > rd_ptr ? fvc_out_wrptr - rd_ptr : buffer_size * 2 + fvc_out_wrptr - rd_ptr;
        }
        return buffer_size - words_occupied;
    }

    inline void init(uint32_t data_buf_start, uint32_t data_buf_size_words, uint32_t ptr_update_addr) {
        uint32_t words = sizeof(fvc_state)/4;
        uint32_t *ptr = (uint32_t *) this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
        chan_num = 1;
        buffer_start = data_buf_start;
        buffer_size = data_buf_size_words;
        remote_buffer_start = data_buf_start + buffer_size * PACKET_WORD_SIZE_BYTES;
        remote_ptr_update_addr = ptr_update_addr;
    }

    inline uint32_t words_before_buffer_wrap(uint32_t ptr) {
        if (ptr >= buffer_size) {
            return buffer_size * 2 - ptr;
        } else {
            return buffer_size - ptr;
        }
    }

    inline uint32_t words_before_local_buffer_wrap() {
        if (fvc_pull_wrptr >= buffer_size) {
            return buffer_size * 2 - fvc_pull_wrptr;
        } else {
            return buffer_size - fvc_pull_wrptr;
        }
    }

    inline uint32_t get_local_buffer_pull_addr() {
        uint32_t addr = buffer_start;
        uint32_t offset = fvc_pull_wrptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline uint32_t get_local_buffer_read_addr() {
        uint32_t addr = buffer_start;
        uint32_t offset = fvc_out_rdptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline uint32_t get_remote_buffer_write_addr() {
        uint32_t addr = remote_buffer_start;
        uint32_t offset = fvc_out_wrptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline void advance_pull_wrptr(uint32_t num_words) {
        uint32_t temp = fvc_pull_wrptr + num_words;
        if (temp >= buffer_size * 2) {
            temp = 0;
        }
        fvc_pull_wrptr = temp;
    }

    inline void advance_out_wrptr(uint32_t num_words) {
        uint32_t temp = fvc_out_wrptr + num_words;
        if (temp >= buffer_size * 2) {
            temp = 0;
        }
        fvc_out_wrptr = temp;
    }

    inline void advance_out_rdptr(uint32_t num_words) {
        uint32_t temp = fvc_out_rdptr + num_words;
        if (temp >= buffer_size * 2) {
            temp = 0;
        }
        fvc_out_rdptr = temp;
    }

    inline void register_pull_data(uint32_t num_words_to_pull) {
        pull_words_in_flight += num_words_to_pull;
        advance_pull_wrptr(num_words_to_pull);
        words_since_last_sync += num_words_to_pull;
        packet_words_remaining -= num_words_to_pull;
        //also check for complete packet pulled.
        if ((packet_words_remaining == 0) or (words_since_last_sync >= FVC_SYNC_THRESHOLD)) {
            sync_buf[sync_buf_wrptr] = fvc_pull_wrptr;
            if (get_num_words_free()) {
                advance_pull_wrptr(1);
                sync_buf_advance_wrptr();
            } else {
                sync_pending = 1;
            }
            words_since_last_sync = 0;
        }
    }

    inline void check_sync_pending() {
        if (sync_pending) {
            if (get_num_words_free()) {
                advance_pull_wrptr(1);
                sync_buf_advance_wrptr();
                sync_pending = 0;
            }
        }
    }

    inline uint32_t forward_data_from_fvc_buffer() {

        uint32_t total_words_to_forward = 0;
        uint32_t wrptr = sync_buf[sync_buf_rdptr];

        total_words_to_forward = wrptr > fvc_out_rdptr ? wrptr - fvc_out_rdptr : buffer_size * 2 + wrptr - fvc_out_rdptr;

        uint32_t remote_fvc_buffer_space = get_remote_num_words_free();
        if (remote_fvc_buffer_space < (total_words_to_forward + 1)) {
            // +1 is for pointer sync word.
            // If fvc receiver buffer on link partner does not have space to receive the
            // full sync buffer entry, we skip and try again next time.
            return 0;
        }

        // Now that there is enough space in receiver buffer we will send total_words_to_forward number of words.
        // This means that we may need to break up the writes to multiple ethernet packets
        // depending on whether local buffer is wrapping, remote buffer is wrapping,
        // we are writing sync word etc.

        uint32_t src_addr = 0;
        uint32_t dest_addr = 0; // should be second half of fvc buffer.
        uint32_t words_remaining = total_words_to_forward;
        while (words_remaining) {
            uint32_t num_words_before_local_wrap = words_before_buffer_wrap(fvc_out_rdptr);
            uint32_t num_words_before_remote_wrap = words_before_buffer_wrap(fvc_out_wrptr);;
            uint32_t words_to_forward = std::min(num_words_before_local_wrap, num_words_before_remote_wrap);
            words_to_forward = std::min(words_to_forward, words_remaining);
            words_to_forward = std::min(words_to_forward, DEFAULT_MAX_ETH_SEND_WORDS);
            src_addr = get_local_buffer_read_addr();
            dest_addr = get_remote_buffer_write_addr();

            internal_::eth_send_packet(0, src_addr/PACKET_WORD_SIZE_BYTES, dest_addr/PACKET_WORD_SIZE_BYTES, words_to_forward);
            advance_out_rdptr(words_to_forward);
            advance_out_wrptr(words_to_forward);
            words_remaining -= words_to_forward;
        }
        // after sending all the data, send the last word which is pointer sync word.
        volatile uint32_t * sync_ptr = (volatile uint32_t *)get_local_buffer_read_addr();
        advance_out_rdptr(1);
        sync_ptr[0] = fvc_out_wrptr;
        sync_ptr[1] = 0;
        sync_ptr[2] = 0;
        sync_ptr[3] = fvc_out_rdptr;
        internal_::eth_send_packet(0, ((uint32_t)sync_ptr)/PACKET_WORD_SIZE_BYTES, remote_ptr_update_addr/PACKET_WORD_SIZE_BYTES, 1);
        sync_buf_advance_rdptr();
        return total_words_to_forward;
    }

    inline void sync_buf_advance_wrptr() {
        sync_buf_wrptr = (sync_buf_wrptr + 1) & SYNC_BUF_PTR_MASK;
    }

    inline void sync_buf_advance_rdptr() {
        sync_buf_rdptr = (sync_buf_rdptr + 1) & SYNC_BUF_PTR_MASK;
    }

    inline bool sync_buf_empty() {
        return (sync_buf_wrptr == sync_buf_rdptr);
    }

    inline bool sync_buf_full() {
        return !sync_buf_empty() && ((sync_buf_wrptr & SYNC_BUF_SIZE_MASK) == (sync_buf_rdptr & SYNC_BUF_SIZE_MASK));
    }

} fvc_state_t;

static_assert(sizeof(fvc_state_t) % 4 == 0);

typedef struct fvc_inbound_state {
    volatile chan_payload_ptr inbound_wrptr;
    volatile chan_payload_ptr inbound_rdptr;
    uint32_t remote_ptr_update_addr;
    uint8_t chan_num;
    uint8_t packet_in_progress;
    uint8_t sync_buf_wrptr;
    uint8_t sync_buf_rdptr;
    uint32_t packet_words_remaining;
    uint32_t packet_words_sent;
    uint32_t fvc_out_wrptr;
    uint32_t fvc_out_rdptr;
    volatile uint32_t fvc_pull_rdptr;
    uint32_t buffer_size;
    uint32_t buffer_start;
    uint32_t pull_words_in_flight;
    uint32_t words_since_last_sync;
    uint32_t words_to_forward;
    uint8_t curr_packet_valid;
    uint8_t padding[3];
    uint32_t sync_buf[SYNC_BUF_SIZE];
    uint64_t packet_dest;
    packet_header_t current_packet_header;

    inline void init(uint32_t data_buf_start, uint32_t data_buf_size_words, uint32_t ptr_update_addr) {
        uint32_t words = sizeof(fvc_inbound_state)/4;
        uint32_t *ptr = (uint32_t *) this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
        chan_num = 1;
        buffer_start = data_buf_start;
        buffer_size = data_buf_size_words;
        remote_ptr_update_addr = ptr_update_addr;
    }

    inline uint32_t inc_ptr_with_wrap(uint32_t ptr, uint32_t inc) {
        uint32_t temp = ptr + inc;
        if (temp >= buffer_size * 2) {
            temp = 0;
        }
        return temp;
    }

    inline void advance_out_wrptr(uint32_t num_words) {
        uint32_t temp = fvc_out_wrptr + num_words;
        if (temp >= buffer_size * 2) {
            temp = 0;
        }
        fvc_out_wrptr = temp;
    }

    inline void advance_out_rdptr(uint32_t num_words) {
        uint32_t temp = fvc_out_rdptr + num_words;
        if (temp >= buffer_size * 2) {
            temp = 0;
        }
        fvc_out_rdptr = temp;
    }

    inline uint32_t words_before_buffer_wrap(uint32_t ptr) {
        if (ptr >= buffer_size) {
            return buffer_size * 2 - ptr;
        } else {
            return buffer_size - ptr;
        }
    }

    inline uint32_t get_num_words_available() const {
        uint32_t wrptr = inbound_wrptr.ptr;
        uint32_t words_occupied = 0;
        if (fvc_out_rdptr != wrptr) {
            words_occupied = wrptr > fvc_out_rdptr ? wrptr - fvc_out_rdptr : buffer_size * 2 + wrptr - fvc_out_rdptr;
        }
        return words_occupied;
    }

    inline bool get_curr_packet_valid() {
        if (!curr_packet_valid && (get_num_words_available() >= PACKET_HEADER_SIZE_WORDS)) {
            // Wait for a full packet header to arrive before advancing to next packet.
            this->advance_next_packet();
        }
        return this->curr_packet_valid;
    }

    inline uint32_t get_local_buffer_read_addr() {
        uint32_t addr = buffer_start;
        uint32_t offset = fvc_out_rdptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline void update_remote_rdptr_sent() {
        if (inbound_wrptr.ptr_cleared != inbound_rdptr.ptr) {
            inbound_rdptr.ptr = inbound_wrptr.ptr_cleared;
            internal_::eth_send_packet(0, ((uint32_t)&inbound_rdptr)/PACKET_WORD_SIZE_BYTES, remote_ptr_update_addr/PACKET_WORD_SIZE_BYTES, 1);
        }
    }

    inline void update_remote_rdptr_cleared() {
        if (fvc_pull_rdptr != inbound_rdptr.ptr_cleared) {
            inbound_rdptr.ptr_cleared = fvc_pull_rdptr;
            internal_::eth_send_packet(0, ((uint32_t)&inbound_rdptr)/PACKET_WORD_SIZE_BYTES, remote_ptr_update_addr/PACKET_WORD_SIZE_BYTES, 1);
        }
    }

    inline void advance_next_packet() {
        if(this->get_num_words_available() >= PACKET_HEADER_SIZE_WORDS) {
            tt_l1_ptr uint32_t* packet_header_ptr = (uint32_t *)&current_packet_header;
            tt_l1_ptr uint32_t* next_header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(get_local_buffer_read_addr());
            uint32_t words_before_wrap = words_before_buffer_wrap(fvc_out_rdptr);
            uint32_t dwords_to_copy = PACKET_HEADER_SIZE_BYTES / 4;
            if (words_before_wrap < PACKET_HEADER_SIZE_WORDS) {
                // Header spans buffer end.
                // Needs to be copied in two steps.
                uint32_t dwords_before_wrap = words_before_wrap * PACKET_WORD_SIZE_BYTES / 4;
                uint32_t dwords_after_wrap = dwords_to_copy - dwords_before_wrap;
                for (uint32_t i = 0; i < dwords_before_wrap; i++) {
                    packet_header_ptr[i] = next_header_ptr[i];
                }
                next_header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(buffer_start);
                for (uint32_t i = 0; i < dwords_after_wrap; i++) {
                    packet_header_ptr[i + dwords_before_wrap] = next_header_ptr[i];
                }
            } else {
                for (uint32_t i = 0; i < dwords_to_copy; i++) {
                    packet_header_ptr[i] = next_header_ptr[i];
                }
            }

            this->packet_words_remaining = (this->current_packet_header.routing.packet_size_bytes + PACKET_WORD_SIZE_BYTES - 1) >> 4;
            this->packet_words_sent = 0;
            this->curr_packet_valid = true;
       }
    }

    inline uint32_t pull_data_from_fvc_buffer() {

        uint32_t words_available = get_num_words_available();
        words_available = std::min(words_available, packet_words_remaining);

        if (packet_in_progress == 0) {
            advance_out_wrptr(words_available);
            pull_request.wr_ptr = fvc_out_wrptr;
            pull_request.rd_ptr = fvc_out_rdptr;//inbound_rdptr.ptr_cleared;
            //fvc_pull_rdptr = fvc_out_rdptr;
            pull_request.size = this->current_packet_header.routing.packet_size_bytes;
            pull_request.buffer_size = buffer_size;
            pull_request.buffer_start = xy_local_addr + buffer_start;
            pull_request.ack_addr = xy_local_addr + (uint32_t)&fvc_pull_rdptr;
            pull_request.flags = FORWARD;
            packet_in_progress = 1;
            packet_words_remaining -= words_available;
            advance_out_rdptr(words_available);
            //issue noc write to noc target of pull request.

        } else {
            if (packet_words_remaining) {
                if (words_available) {
                    advance_out_wrptr(words_available);
                    //pull_request.wr_ptr = inc_ptr_with_wrap(pull_request.wr_ptr, words_available);
                    //set packet_dest to request q entry + pull_request.wr_ptr.
                    noc_inline_dw_write(packet_dest, fvc_out_wrptr);
                    advance_out_rdptr(words_available);
                    packet_words_remaining -= words_available;
                }
            } else if (fvc_pull_rdptr == fvc_out_rdptr) {
                // all data has been pulled and cleared from local buffer
                packet_in_progress = 0;
                curr_packet_valid = 0;
            }

        }
        // send ptr cleared to ethernet sender.
        update_remote_rdptr_cleared();
        return words_available;
    }

    inline void issue_async_write() {
        uint32_t words_available = get_num_words_available();
        words_available = std::min(words_available, packet_words_remaining);
        words_available = std::min(words_available, words_before_buffer_wrap(fvc_out_rdptr));
        if (words_available) {
            noc_async_write(get_local_buffer_read_addr(), packet_dest, words_available*PACKET_WORD_SIZE_BYTES);
            packet_words_remaining -= words_available;
            advance_out_wrptr(words_available);
            advance_out_rdptr(words_available);
            packet_dest += words_available*PACKET_WORD_SIZE_BYTES;
        }
    }

    inline void process_inbound_packet() {
        if (current_packet_header.routing.flags == FORWARD && current_packet_header.session.command == ASYNC_WR) {
            if (packet_in_progress == 0) {
                packet_dest = ((uint64_t)current_packet_header.session.target_offset_h << 32) | current_packet_header.session.target_offset_l;
                packet_words_remaining -= PACKET_HEADER_SIZE_WORDS;
                advance_out_wrptr(PACKET_HEADER_SIZE_WORDS);
                advance_out_rdptr(PACKET_HEADER_SIZE_WORDS);
                // subtract the header words. Remaining words are the data to be written to packet_dest.
                // Remember to account for trailing bytes which may not be a full packet word.
                packet_in_progress = 1;
                issue_async_write();
            } else {
                flush_async_writes();
                if (packet_words_remaining) {
                    issue_async_write();
                } else {
                    packet_in_progress = 0;
                    curr_packet_valid = 0;
                }

            }
        } else {
            //pull_data_from_fvc_buffer();
        }
    }

    inline void flush_async_writes() {
        noc_async_write_barrier();
        fvc_pull_rdptr = fvc_out_rdptr;
        update_remote_rdptr_cleared();
    }

} fvc_inbound_state_t;

typedef struct router_state {
    uint32_t sync_in;
    uint32_t padding_in[3];
    uint32_t sync_out;
    uint32_t padding_out[3];
    uint32_t scratch[4];
} router_state_t;

inline uint64_t get_timestamp() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

inline uint64_t get_timestamp_32b() {
    return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

void zero_l1_buf(tt_l1_ptr uint32_t* buf, uint32_t size_bytes) {
    for (uint32_t i = 0; i < size_bytes/4; i++) {
        buf[i] = 0;
    }
}

static FORCE_INLINE
void write_test_results(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

static FORCE_INLINE
void write_kernel_status(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

static FORCE_INLINE
void set_64b_result(uint32_t* buf, uint64_t val, uint32_t index = 0) {
    if (buf != nullptr) {
        buf[index] = val >> 32;
        buf[index+1] = val & 0xFFFFFFFF;
    }
}

inline bool req_buf_ptrs_empty(uint32_t wrptr, uint32_t rdptr) {
  return (wrptr == rdptr);
}

inline bool req_buf_ptrs_full(uint32_t wrptr, uint32_t rdptr) {
  uint32_t distance = wrptr >= rdptr ? wrptr - rdptr : wrptr + 2 * CHAN_REQ_BUF_SIZE - rdptr;
  return !req_buf_ptrs_empty(wrptr, rdptr) && (distance >= CHAN_REQ_BUF_SIZE);
}

inline bool fvc_req_buf_is_empty(const volatile chan_req_buf* req_buf) {
  return req_buf_ptrs_empty(req_buf->wrptr.ptr, req_buf->rdptr.ptr);
}

inline bool fvc_req_buf_is_full(const volatile chan_req_buf* req_buf) {
  return req_buf_ptrs_full(req_buf->wrptr.ptr, req_buf->rdptr.ptr);
}

inline bool fvc_req_valid(const volatile chan_req_buf* req_buf) {
  uint32_t rd_index = req_buf->rdptr.ptr & CHAN_REQ_BUF_SIZE_MASK;
  return req_buf->chan_req[rd_index].pull_request.flags != 0;
}

inline uint32_t num_words_available_to_pull(volatile pull_request_t *pull_request) {

    uint32_t wr_ptr = pull_request->wr_ptr;
    uint32_t rd_ptr = pull_request->rd_ptr;
    uint32_t buf_size = pull_request->buffer_size;

    if (wr_ptr == rd_ptr) {
        //buffer empty.
        return 0;
    }
    uint32_t num_words = wr_ptr > rd_ptr ? wr_ptr - rd_ptr : buf_size * 2 + wr_ptr - rd_ptr;

    //num_words = std::min(num_words, this->get_curr_packet_words_remaining());
    return num_words;
}

inline uint32_t advance_ptr(uint32_t buffer_size, uint32_t ptr, uint32_t inc_words) {
    uint32_t temp = ptr + inc_words;
    if (temp >= buffer_size * 2) {
        temp = 0;
    }
    return temp;
}

inline uint32_t words_before_buffer_wrap(uint32_t buffer_size, uint32_t rd_ptr) {
    if (rd_ptr >= buffer_size) {
        return buffer_size * 2 - rd_ptr;
    } else {
        return buffer_size - rd_ptr;
    }
}

inline uint32_t get_rd_ptr_offset_words(pull_request_t *pull_request) {
    uint32_t offset = pull_request->rd_ptr;
    if (pull_request->rd_ptr >= pull_request->buffer_size) {
        offset -= pull_request->buffer_size;
    }
    return offset;
}

inline void update_pull_request_words_cleared(pull_request_t *pull_request) {
    noc_inline_dw_write(pull_request->ack_addr, pull_request->rd_ptr);
}

inline uint32_t get_num_words_to_pull(volatile pull_request_t *pull_request, fvc_state_t *fvc_state) {


    uint32_t num_words_to_pull = num_words_available_to_pull(pull_request);
    uint32_t num_words_before_wrap = words_before_buffer_wrap(pull_request->buffer_size, pull_request->rd_ptr);

    num_words_to_pull = std::min(num_words_to_pull, num_words_before_wrap);
    uint32_t fvc_buffer_space = fvc_state->get_num_words_free();
    num_words_to_pull = std::min(num_words_to_pull, fvc_buffer_space);

    if (num_words_to_pull == 0) {
        return 0;
    }

    uint32_t fvc_space_before_wptr_wrap = fvc_state->words_before_local_buffer_wrap();
    num_words_to_pull = std::min(num_words_to_pull, fvc_space_before_wptr_wrap);
    num_words_to_pull = std::min(num_words_to_pull, DEFAULT_MAX_NOC_SEND_WORDS);

    return num_words_to_pull;
}


inline uint32_t pull_data_to_fvc_buffer(volatile pull_request_t *pull_request, fvc_state_t *fvc_state) {


    if (fvc_state->packet_in_progress == 0) {
        uint32_t size = pull_request->size;
        fvc_state->packet_words_remaining = (size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
        fvc_state->packet_in_progress = 1;
    }

    uint32_t num_words_to_pull = get_num_words_to_pull(pull_request, fvc_state);
    bool full_packet_sent = (num_words_to_pull == fvc_state->packet_words_remaining);
    if (num_words_to_pull == 0) {
        return 0;
    }

    uint32_t rd_offset = get_rd_ptr_offset_words((pull_request_t *)pull_request);
    uint64_t src_addr = pull_request->buffer_start + (rd_offset * PACKET_WORD_SIZE_BYTES);
    uint32_t fvc_addr = fvc_state->get_local_buffer_pull_addr();

    //pull_data_from_remote();
    noc_async_read(src_addr, fvc_addr, num_words_to_pull * PACKET_WORD_SIZE_BYTES);
    fvc_state->register_pull_data(num_words_to_pull);
    pull_request->rd_ptr = advance_ptr(pull_request->buffer_size, pull_request->rd_ptr, num_words_to_pull);

    //TODO: this->remote_wptr_update(num_words_to_forward);

    return num_words_to_pull;
}



/**
 *  Polling for ready signal from the remote peers of all input and output queues.
 *  Blocks until all are ready, but doesn't block polling on each individual queue.
 *  Returns false in case of timeout.
 */
bool wait_all_src_dest_ready(volatile router_state_t* router_state, uint32_t timeout_cycles = 0) {

    bool src_ready = false;
    bool dest_ready = false;

    uint32_t iters = 0;

    uint32_t start_timestamp = get_timestamp_32b();
    uint32_t sync_in_addr = ((uint32_t)&router_state->sync_in)/PACKET_WORD_SIZE_BYTES;
    uint32_t sync_out_addr = ((uint32_t)&router_state->sync_out)/PACKET_WORD_SIZE_BYTES;

    uint32_t scratch_addr = ((uint32_t)&router_state->scratch)/PACKET_WORD_SIZE_BYTES;
    router_state->scratch[0] = 0xAA;
    //send_buf[1] = 0x0;
    //send_buf[2] = 0x0;
    //send_buf[3] = 0x0;

    while (!src_ready or !dest_ready) {
        if (router_state->sync_out != 0xAA) {
            internal_::eth_send_packet(0, scratch_addr, sync_in_addr, 1);
        } else {
            dest_ready = true;
        }

        if (!src_ready && router_state->sync_in == 0xAA) {
            internal_::eth_send_packet(0, sync_in_addr, sync_out_addr, 1);
            src_ready = true;
        }

        iters++;
        if (timeout_cycles > 0) {
            uint32_t cycles_since_start = get_timestamp_32b() - start_timestamp;
            if (cycles_since_start > timeout_cycles) {
                return false;
            }
        }

#if defined(COMPILE_FOR_ERISC)
        if ((timeout_cycles == 0) && (iters & 0xFFF) == 0) {
            //if timeout is disabled, context switch every 4096 iterations.
            //this is necessary to allow ethernet routing layer to operate.
            //this core may have pending ethernet routing work.
            internal_::risc_context_switch();
        }
#endif
    }
    return true;
}
