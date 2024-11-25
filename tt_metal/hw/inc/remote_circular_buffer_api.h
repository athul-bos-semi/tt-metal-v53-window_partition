// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

namespace experimental {


FORCE_INLINE void remote_cb_wait_front(uint32_t cb_id, uint32_t num_pages) {
    const RemoteReceiverCBInterface &remote_cb = get_remote_receiver_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * remote_cb.fifo_page_size;
    uint32_t num_pages_wait = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    uint32_t num_pages_recv = 0;
    uint32_t pages_acked = 0;
    uint32_t pages_sent = 0;

    volatile tt_l1_ptr uint32_t *pages_acked_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(remote_cb.aligned_pages_acked_ptr);
    volatile tt_l1_ptr uint32_t *pages_sent_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(remote_cb.aligned_pages_sent_ptr);
    do {

        pages_acked = *pages_acked_ptr;
        pages_sent = *pages_sent_ptr;
        num_pages_recv = pages_sent - pages_acked;
    } while (num_pages_recv < num_pages_wait);
}


FORCE_INLINE void remote_cb_pop_front(uint32_t cb_id, uint32_t num_pages, uint8_t noc = noc_index) {
    RemoteReceiverCBInterface &remote_cb = get_remote_receiver_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * remote_cb.fifo_page_size;
    uint32_t num_aligned_pages = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;

    volatile tt_l1_ptr uint32_t *pages_acked_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(remote_cb.aligned_pages_acked_ptr);

    *pages_acked_ptr += num_aligned_pages;
    remote_cb.fifo_rd_ptr += len_bytes;

    if (remote_cb.fifo_rd_ptr >= remote_cb.fifo_limit_page_aligned) {
        remote_cb.fifo_rd_ptr = remote_cb.fifo_start_addr;
    }

    uint64_t remote_ack_ptr_addr = get_noc_addr(remote_cb.sender_noc_x, remote_cb.sender_noc_y, (uint32_t)pages_acked_ptr, noc);
    noc_semaphore_inc(remote_ack_ptr_addr, num_aligned_pages, noc);
}

FORCE_INLINE void remote_cb_reserve_back(uint32_t cb_id, uint32_t num_pages) {
    const RemoteSenderCBInterface &remote_cb = get_remote_sender_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * remote_cb.fifo_page_size;
    uint32_t num_pages_wait = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    uint32_t free_pages;

    volatile tt_l1_ptr uint32_t *pages_sent_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(remote_cb.aligned_pages_sent_ptr);
    volatile tt_l1_ptr uint32_t *pages_acked_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(remote_cb.aligned_pages_sent_ptr + remote_cb.num_receivers * L1_ALIGNMENT);

    uint32_t num_receivers = remote_cb.num_receivers;
    uint32_t fifo_aligned_num_pages = remote_cb.fifo_aligned_num_pages;
    for (uint32_t i=0; i < num_receivers; ++i) {
        do {
            uint32_t pages_acked = *pages_acked_ptr;
            uint32_t pages_sent = *pages_sent_ptr;
            free_pages = fifo_aligned_num_pages - (pages_sent - pages_acked);
        } while (free_pages < num_pages_wait);
        pages_acked_ptr += L1_ALIGNMENT / sizeof(uint32_t);
        pages_sent_ptr += L1_ALIGNMENT / sizeof(uint32_t);
    }
}

FORCE_INLINE void remote_cb_push_back_and_write_pages(uint32_t cb_id, uint32_t local_cb_addr, uint32_t num_pages, uint32_t num_rows, uint32_t coalesced_num_pages_per_row, uint32_t coalesced_page_size, uint8_t noc = noc_index) {
    RemoteSenderCBInterface &remote_cb = get_remote_sender_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * remote_cb.fifo_page_size;
    uint32_t pages_sent = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    uint32_t num_receivers = remote_cb.num_receivers;

    uint32_t next_receiver_start_addr_stride = coalesced_num_pages_per_row * coalesced_page_size;
    uint32_t next_block_row_stride = next_receiver_start_addr_stride * num_receivers;

    uint32_t dest_addr;

    uint32_t next_receiver_start_addr_offset = 0;
    volatile tt_l1_ptr uint32_t *pages_sent_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(remote_cb.aligned_pages_sent_ptr);
    volatile tt_l1_ptr uint32_t *remote_noc_xy_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(remote_cb.receiver_noc_xy_ptr);
    for (uint32_t i=0; i < num_receivers; ++i) {

        uint32_t src_addr = local_cb_addr + next_receiver_start_addr_offset;
        dest_addr = remote_cb.fifo_wr_ptr;

        uint32_t remote_noc_xy = uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, remote_noc_xy_ptr[0]), DYNAMIC_NOC_Y(noc, remote_noc_xy_ptr[1])));
        uint64_t dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

        noc_async_write_one_packet_set_state(dest_noc_addr, coalesced_page_size, noc);

        for (uint32_t h = 0; h < num_rows; ++h) {
            uint32_t prev_src_addr = src_addr;
            for (uint32_t w = 0; w < coalesced_num_pages_per_row; ++w) {
                dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

                if ((dest_addr + coalesced_page_size) > remote_cb.fifo_limit_page_aligned) {

                    uint32_t first_len_bytes = remote_cb.fifo_limit_page_aligned - dest_addr;
                    uint32_t second_len_bytes = coalesced_page_size - first_len_bytes;

                    if (first_len_bytes != 0) {
                        noc_async_write_one_packet(src_addr, dest_noc_addr, first_len_bytes, noc);
                        src_addr += first_len_bytes;
                    }

                    dest_addr = remote_cb.fifo_start_addr;
                    dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

                    noc_async_write_one_packet(src_addr, dest_noc_addr, second_len_bytes, noc);

                    src_addr += second_len_bytes;
                    dest_addr += second_len_bytes;
                    dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

                    noc_async_write_one_packet_set_state(dest_noc_addr, coalesced_page_size, noc);

                } else {
                    noc_async_write_one_packet_with_state(src_addr, dest_noc_addr, noc);

                    src_addr += coalesced_page_size;
                    dest_addr += coalesced_page_size;
                }
            }
            src_addr = prev_src_addr + next_block_row_stride;
        }
        next_receiver_start_addr_offset += next_receiver_start_addr_stride;
        *pages_sent_ptr += pages_sent;

        uint64_t remote_sent_ptr_addr = get_noc_addr_helper(remote_noc_xy, (uint32_t)pages_sent_ptr);
        noc_semaphore_inc(remote_sent_ptr_addr, pages_sent, noc);
        pages_sent_ptr += L1_ALIGNMENT / sizeof(uint32_t);
        remote_noc_xy_ptr += 2;
    }

    remote_cb.fifo_wr_ptr = dest_addr;

}

}
