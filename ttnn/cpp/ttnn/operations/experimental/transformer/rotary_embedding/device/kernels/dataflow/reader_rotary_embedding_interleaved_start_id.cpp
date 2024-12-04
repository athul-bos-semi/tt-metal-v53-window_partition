// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

// #define DEBUG_READER 1

#ifdef DEBUG_READER
inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint8_t r = 0; r < 32; ++ r) {
        if (r % 8 == 0) {
            DPRINT << ENDL();
        }
        SliceRange sr = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ":  " << TileSlice(cb_id, tile_id, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR, true, untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}
#endif

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t sin_addr = get_arg_val<uint32_t>(2);
    uint32_t num_rows = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);
    uint32_t start_row_id = get_arg_val<uint32_t>(5);
    uint32_t cos_sin_start_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(3);
    constexpr bool input_is_dram = get_compile_time_arg_val(5) == 1;
    constexpr bool sin_is_dram = get_compile_time_arg_val(7) == 1;
    constexpr uint16_t scalar_value = get_compile_time_arg_val(8);
    constexpr uint32_t Ht = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t HtWt = get_compile_time_arg_val(11);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(12);

    constexpr uint32_t onetile = 1;
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const DataFormat input_data_format = get_dataformat(input_cb_id);

    const InterleavedAddrGenFast<input_is_dram> s0 = {
        .bank_base_address = src_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const DataFormat sin_data_format = get_dataformat(sin_cb_id);

    const InterleavedAddrGenFast<sin_is_dram> s2 = {
        .bank_base_address = sin_addr, .page_size = sin_tile_bytes, .data_format = sin_data_format};

#ifdef DEBUG_READER
    DPRINT << "Input DF " << (uint32_t)input_data_format << " Input Tsz " << input_tile_bytes << ENDL();
    DPRINT << "Sine DF " << (uint32_t)sin_data_format << " Sine Tsz " << sin_tile_bytes << ENDL();
    DPRINT << "num_rows " << num_rows << " Wt " << Wt << " start_id " << start_id << ENDL();
    DPRINT << "start_row_id " << start_row_id << " cos_sin_start_id " << cos_sin_start_id << ENDL();
#endif

    uint32_t input_curr_id = start_id;
    uint32_t cos_sin_curr_id = cos_sin_start_id;
    uint32_t ht = start_row_id;

    cb_reserve_back(sin_cb_id, Wt);
    uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);
    for (uint32_t i = 0; i < Wt; i++) {
        noc_async_read_tile(cos_sin_curr_id, s2, sin_l1_write_addr);
        cos_sin_curr_id++;
        sin_l1_write_addr += sin_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(sin_cb_id, Wt);

#ifdef DEBUG_READER
    cb_wait_front(sin_cb_id, Wt);
    print_full_tile(sin_cb_id, 0, false);
    print_full_tile(sin_cb_id, 1, false);
#endif

    /*
        // read a ublock of tiles from src to CB, and then push the ublock to unpacker
        for (uint32_t i = 0; i < num_rows; ++i) {
            for (uint32_t j = 0; j < Wt; ++j) {
                cb_reserve_back(input_cb_id, onetile);
                uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
                noc_async_read_tile(input_curr_id, s0, input_l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(input_cb_id, onetile);
                input_curr_id++;
            }
        }
    */
}
