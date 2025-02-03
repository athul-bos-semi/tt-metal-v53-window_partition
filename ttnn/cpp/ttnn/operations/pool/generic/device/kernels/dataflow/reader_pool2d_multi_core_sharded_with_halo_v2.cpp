// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include "dataflow_api.h"

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

#define ALWI inline __attribute__((always_inline))

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n / 2; ++i) {
        ptr[i] = (val | (val << 16));
    }
    return true;
}

/**
 * Pool 2D.
 */
void kernel_main() {
    const uint32_t reader_nindices = get_compile_time_arg_val(0);
    const uint32_t window_h = get_compile_time_arg_val(1);
    const uint32_t window_w = get_compile_time_arg_val(2);

    const int32_t pad_w = get_compile_time_arg_val(3);

    // channel size in bytes
    const uint32_t in_nbytes_c = get_compile_time_arg_val(4);

    // input tensor height / width / channels
    const int32_t in_w = get_compile_time_arg_val(5);
    const uint32_t in_cb_nsticks = get_compile_time_arg_val(6);

    const uint32_t in_c = get_compile_time_arg_val(7);

    const uint32_t split_reader = get_compile_time_arg_val(9);
    const uint32_t reader_id = get_compile_time_arg_val(10);

    // bf16 value packed in u32
    constexpr uint16_t bf16_scalar = get_compile_time_arg_val(11);

    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(13);

    constexpr uint32_t ceil_pad_w = get_compile_time_arg_val(15);

    constexpr uint32_t TILE_WIDTH = 32;

    constexpr uint32_t in_cb_id = (reader_id == 1) ? tt::CBIndex::c_1 : tt::CBIndex::c_0;
    constexpr uint32_t in_shard_cb_id = tt::CBIndex::c_2;  // local input shard
    constexpr uint32_t in_reader_indices_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in_scalar_cb_id = tt::CBIndex::c_4;

    constexpr uint32_t ROW_HW = 64;

    // Reduce scalar = 1
    if (reader_id == 0) {
        cb_reserve_back(in_scalar_cb_id, 1);

        // fill 1 row w/ scalar
        fill_with_val(get_write_ptr(in_scalar_cb_id), ROW_HW, bf16_scalar);
        cb_push_back(in_scalar_cb_id, 1);
    }

    uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
    volatile tt_l1_ptr uint16_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(reader_indices_l1_addr);

    uint32_t in_w_padded = in_w + 2 * pad_w + ceil_pad_w;

    uint32_t npages_to_reserve = 1;
    uint32_t counter = reader_id;
    while (counter < reader_nindices) {
        cb_reserve_back(in_cb_id, npages_to_reserve);
        uint32_t out_l1_write_addr_base = get_write_ptr(in_cb_id);
        uint32_t out_l1_write_addr = out_l1_write_addr_base;
        uint16_t top_left_local_index = reader_indices_ptr[counter++];
        uint32_t h_multiples = 0;
        for (uint32_t h = 0; h < window_h; ++h, h_multiples += in_w_padded) {
            uint32_t stick_offset = top_left_local_index + h_multiples;
            uint32_t read_offset = in_l1_read_base_addr + (stick_offset * in_nbytes_c);
            noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, in_nbytes_c * window_w);
            out_l1_write_addr += in_nbytes_c * window_w;
        }
        if (split_reader) {
            counter++;  // interleave the indices
        }
        noc_async_read_barrier();
        cb_push_back(in_cb_id, npages_to_reserve);
    }
}  // kernel_main()
