// SPDX-FileCopyrightText: © 2025 BOS
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb = get_compile_time_arg_val(1);

    constexpr uint32_t stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t stride_size = get_compile_time_arg_val(3);
    constexpr uint32_t window_row_sticks_size = get_compile_time_arg_val(4);

    constexpr uint32_t window_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_windows = get_compile_time_arg_val(6);
    constexpr uint32_t num_output_sticks = get_compile_time_arg_val(7);

    uint32_t l1_write_addr = get_write_ptr(output_cb);
    const uint32_t base_l1_read_addr = get_read_ptr(input_cb);
    const uint64_t noc_addr = get_noc_addr(base_l1_read_addr);
    noc_async_read_one_packet_set_state(noc_addr, num_output_sticks * stick_size);

    uint32_t current_window_addr = base_l1_read_addr;
    for (uint32_t current_window = 0; current_window < num_windows; current_window++) {
        uint32_t current_row_start = current_window_addr;
        for (uint32_t current_window_along_height = 0; current_window_along_height < window_size; current_window_along_height++) {
            for (uint32_t current_window_along_width = 0; current_window_along_width < window_size; current_window_along_width++) {
                noc_async_read_one_packet_with_state<true>(current_row_start + current_window_along_width * stick_size, l1_write_addr);
                l1_write_addr += stick_size;
            }
            current_row_start += ((num_windows - 1) * window_row_sticks_size);
        }
        current_window_addr += window_row_sticks_size;
    }

    noc_async_read_barrier();
}