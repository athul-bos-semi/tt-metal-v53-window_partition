// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void transpose(uint32_t cb_in, uint32_t cb_out) {
    cb_wait_front(cb_in, ONE_TILE);

    tile_regs_acquire();
    tile_regs_wait();

    transpose_wh_init_short(cb_in);
    transpose_wh_tile(cb_in, 0, 0);

    cb_reserve_back(cb_out, ONE_TILE);
    pack_tile(0, cb_out);

    tile_regs_commit();
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t input0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t input1_cb = get_compile_time_arg_val(1);
    constexpr uint32_t input0_transpose_cb = get_compile_time_arg_val(2);
    constexpr uint32_t input1_transpose_cb = get_compile_time_arg_val(3);
    constexpr uint32_t concat_cb = get_compile_time_arg_val(4);
    constexpr uint32_t output_transpose_cb = get_compile_time_arg_val(5);
    constexpr uint32_t output_cb = get_compile_time_arg_val(6);

    constexpr uint32_t input0_num_tiles_height = get_compile_time_arg_val(7);
    constexpr uint32_t input0_num_tiles_width = get_compile_time_arg_val(8);
    constexpr uint32_t input1_num_tiles_height = get_compile_time_arg_val(9);
    constexpr uint32_t input1_num_tiles_width = get_compile_time_arg_val(10);

    constexpr uint32_t tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t groups = get_compile_time_arg_val(12);

    transpose_wh_init(input0_cb, input0_transpose_cb);

    constexpr uint32_t output_num_tiles_width = input0_num_tiles_width + input1_num_tiles_width;

    for (uint32_t i = 0; i < input0_num_tiles_height; i++) {
        reconfig_data_format_srca(input0_cb);
        pack_reconfig_data_format(input0_transpose_cb);
        for (uint32_t j = 0; j < input0_num_tiles_width; j++) {
            transpose(input0_cb, input0_transpose_cb);
        }
        for (uint32_t j = 0; j < input1_num_tiles_width; j++) {
            transpose(input1_cb, input1_transpose_cb);
        }
        reconfig_data_format_srca(concat_cb);
        pack_reconfig_data_format(output_transpose_cb);
        for (uint32_t j = 0; j < output_num_tiles_width; j++) {
            transpose(concat_cb, output_transpose_cb);
        }
    }
}
}  // namespace NAMESPACE
