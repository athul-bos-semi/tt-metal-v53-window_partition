// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>
#include "dataflow_api.h"

inline __attribute__((always_inline)) void fill_pad_cb_with_val(
    const uint32_t cb_id, const uint32_t num_bytes, const uint32_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    for (uint32_t i = 0; i < num_bytes / 2; ++i) {
        ptr[i] = val;
    }
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(1);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t front_pad_n = get_arg_val<uint32_t>(4);
    uint32_t front_pad_c = get_arg_val<uint32_t>(5);
    uint32_t front_pad_h = get_arg_val<uint32_t>(6);
    tt_l1_ptr uint32_t* start_dim_offset = (tt_l1_ptr uint32_t*)(get_arg_addr(7));

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t N_padded = get_compile_time_arg_val(5);
    constexpr uint32_t H_padded = get_compile_time_arg_val(6);
    constexpr uint32_t C_padded = get_compile_time_arg_val(7);
    constexpr uint32_t stick_size_padded = get_compile_time_arg_val(8);
    constexpr uint32_t stick_size_padded_front = get_compile_time_arg_val(9);
    constexpr uint32_t stick_size_padded_end = get_compile_time_arg_val(10);
    constexpr uint32_t num_zero_pad_sticks_read = get_compile_time_arg_val(11);
    constexpr uint32_t last_zero_stick_size = get_compile_time_arg_val(12);
    constexpr uint32_t stick_size_padded_aligned = get_compile_time_arg_val(21);

#define not_pad_by_zero get_compile_time_arg_val(13) == 1
#define front_padding get_compile_time_arg_val(9)
#define unaligned get_compile_time_arg_val(22) == 1

#if (not_pad_by_zero)
    constexpr uint32_t packed_pad_value = get_compile_time_arg_val(14);
    constexpr uint32_t row_major_min_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t num_front_pad_sticks_read = get_compile_time_arg_val(16);
    constexpr uint32_t num_end_pad_sticks_read = get_compile_time_arg_val(17);
    constexpr uint32_t num_sticks_padded_read = get_compile_time_arg_val(18);
#endif

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_pad = tt::CBIndex::c_1;
    constexpr uint32_t cb_pad_align = tt::CBIndex::c_2;

#define stick_size_is_pow2 get_compile_time_arg_val(19) == 1
#if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(20);
#else
    constexpr uint32_t page_size = get_compile_time_arg_val(20);
#endif
#if (stick_size_is_pow2)
    const InterleavedPow2AddrGen<src_is_dram> s = {
        .bank_base_address = src_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
    const InterleavedAddrGen<src_is_dram> s = {.bank_base_address = src_addr, .page_size = page_size};
#endif

    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    uint64_t pad_val_addr = get_read_ptr(cb_pad);
    uint64_t pad_val_noc_addr = get_noc_addr(pad_val_addr);

    uint64_t pad_align_addr = get_read_ptr(cb_pad_align);
    uint64_t pad_align_write_addr = get_write_ptr(cb_pad_align);
    uint64_t pad_align_noc_addr = get_noc_addr(pad_align_addr);

#if (not_pad_by_zero)
    fill_pad_cb_with_val(cb_pad, stick_size_padded, packed_pad_value);
#else
    fill_pad_cb_with_val(cb_pad, stick_size_padded, 0);
#endif

    uint32_t i_stick = start_id;
    uint32_t curr_c = start_dim_offset[2], curr_h = start_dim_offset[1], curr_n = start_dim_offset[3];
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb_reserve_back(cb_in0, num_read_per_barrier);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            bool read_stick = (curr_h >= front_pad_h and curr_h < H) and (curr_c >= front_pad_c and curr_c < C) and
                              (curr_n >= front_pad_n and curr_n < N);
            uint64_t read_noc_addr = get_noc_addr(i_stick, s);
            noc_async_read(pad_val_noc_addr, l1_write_addr, stick_size_padded);

            if (read_stick) {
#if (front_padding)
                // Read noc into cb_pad_align l1
                noc_async_read(read_noc_addr, get_write_ptr(cb_pad_align), stick_size_bytes);
                noc_async_read_barrier();
                memmove(
                    (void*)(l1_write_addr + stick_size_padded_front),
                    (void*)(get_read_ptr(cb_pad_align)),
                    (size_t)(stick_size_bytes));
#else
#if (unaligned)
                noc_async_read(read_noc_addr, get_write_ptr(cb_pad_align), stick_size_bytes);
                noc_async_read_barrier();
                noc_async_read(pad_align_noc_addr, l1_write_addr, stick_size_bytes);
#else
                noc_async_read(read_noc_addr, l1_write_addr, stick_size_bytes);
#endif
#endif
                i_stick++;
            }
            l1_write_addr += stick_size_padded_aligned;
            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, num_read_per_barrier);
    }
}
