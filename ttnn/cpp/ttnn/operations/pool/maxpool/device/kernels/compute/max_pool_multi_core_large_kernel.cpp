// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// #include "compute_kernel_api.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"
// #include "tools/profiler/kernel_profiler.hpp"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
// #include "debug_macros.h"
#define dump_unpack(a) \
    do { DPRINT_UNPACK(DPRINT << "UP: "<< #a " = " << a << ENDL()); } while(false)
#define dump_pack(a) \
    do { DPRINT_PACK(DPRINT << "P: "<< #a " = " << a << ENDL()); } while(false)
#define dump_math(a) \
    do { DPRINT_MATH(DPRINT << "M: "<< #a " = " << a << ENDL()); } while(false)
// SliceRange srt = SliceRange{.h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 4};
// SliceRange srr = SliceRange{.h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
// SliceRange srr1 = SliceRange{.h0 = 1, .h1 = 2, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
// SliceRange src = SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};

// inline void print_tile_rows(uint32_t cb_id, uint32_t rows = 32, uint32_t tile_id = 0, bool untilize = false) {
//     // UNPACK(( DPRINT << "======" << ENDL() ));
//     for (uint16_t r = 0; r < rows; ++r) {
//         SliceRange sr = SliceRange{.h0 = r, .h1 = (uint16_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
//         // UNPACK(( DPRINT << (uint)r << " :: " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() ));
//         UNPACK((DPRINT << (uint)r << " :: " << TileSlice(cb_id, tile_id, sr, true, untilize)));
//     }
//     // UNPACK(( DPRINT << "++++++" << ENDL() ));
// }

// inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
//     UNPACK((DPRINT << "======" << ENDL()));
//     for (uint16_t r = 0; r < 32; ++r) {
//         SliceRange sr = SliceRange{.h0 = r, .h1 = (uint16_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
//         UNPACK((DPRINT << (uint)r << " : " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL()));
//     }
//     UNPACK((DPRINT << "++++++" << ENDL()));
// }

// inline void print_cb_details(uint32_t cb_id) {
//     DPRINT << "cb_id " << cb_id << ": { "
//             << "size: " << cb_interface[cb_id].fifo_size << ", "
//             << "limit: " << cb_interface[cb_id].fifo_limit << ", "
//             << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
//             << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
//             << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
//             << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
//             << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL();
// }
#endif

template <
    uint32_t in_ntiles_hw,
    uint32_t in_ntiles_c,
    uint32_t out_ntiles_c,
    uint32_t in_nblocks_c,
    bool is_partial_tile,
    uint32_t split_reader>
inline void reduce_h_fused(
    const uint32_t in_cb_id,
    const uint32_t in_scalar_cb_id,
    const uint32_t in_stick_index,
    const uint32_t out_cb_id,
    const uint32_t unpA_face_r_dim) {
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;
    const uint32_t curr_in_cb_id =
        split_reader ? (in_cb_id + (in_stick_index & 0x1)) : in_cb_id;
    cb_wait_front(curr_in_cb_id, 1);
    unpack_tilizeA_B_block(
        curr_in_cb_id,
        in_scalar_cb_id,
        in_ntiles_c / in_nblocks_c,
        0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
        num_faces_in_tile /* unpack 1 or 2 faces ) */,
        unpA_face_r_dim);
    //print_full_tile(curr_in_cb_id);
    for (uint32_t c_i = 0; c_i < in_ntiles_c / in_nblocks_c; ++c_i) {
        reduce_tile_math(c_i, num_faces_in_tile /* reduce 1 or 2 faces */);
    }
    cb_pop_front(curr_in_cb_id, 1);
}

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet.
    constexpr uint32_t in_ntiles_hw = get_compile_time_arg_val(0);
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    constexpr uint32_t in_ntiles_hwc = get_compile_time_arg_val(2);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(3);
    constexpr uint32_t out_h = get_compile_time_arg_val(4);
    constexpr uint32_t out_w = get_compile_time_arg_val(5);
    constexpr uint32_t out_ntiles_c = get_compile_time_arg_val(7);
    constexpr uint32_t nblocks = get_compile_time_arg_val(8);

    constexpr uint32_t split_reader = get_compile_time_arg_val(12);

    constexpr uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(13);
    constexpr uint32_t in_c = get_compile_time_arg_val(14);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(15);

    constexpr uint32_t num_output_tiles = out_ntiles_c * nblocks;

    constexpr uint32_t in_cb_id = tt::CB::c_in0;  // and tt::CB::c_in1 for split reader
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in4;
    constexpr uint32_t in_tiled_cb_id = tt::CB::c_intermed0;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;
    constexpr uint32_t interm_reduction_cb_id = tt::CB::c_intermed1;

    constexpr bool is_partial_tile = in_c < 32;
    static_assert((!is_partial_tile || (in_c == 16)), "Partial tile must have c_dim 16");
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;
    constexpr uint32_t MAX_ROWS_FOR_REDUCTION = 16;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    dump_unpack(in_ntiles_hw);
    dump_unpack(in_ntiles_c);
    dump_unpack(in_ntiles_hwc);
    dump_unpack(window_size_hw);
    dump_unpack(out_h);
    dump_unpack(out_w);
    dump_unpack(out_ntiles_c);
    dump_unpack(nblocks);
    dump_unpack(split_reader);
    dump_unpack(nsticks_per_core_by_nblocks);
    dump_unpack(in_c);
    uint32_t num_8_tiles_blocks = 1;
    if (num_output_tiles > MAX_TILES_PER_REDUCTION) {
        num_8_tiles_blocks =
            num_output_tiles / MAX_TILES_PER_REDUCTION;  // For now, only pow of 2 number of channels are supported.
    }
    DPRINT << "in_ntiles_c: " << in_ntiles_c << ENDL();
    DPRINT << "in_nblocks_c: " << in_nblocks_c << ENDL();

    tilizeA_B_reduce_init<true>(
        in_cb_id,
        in_scalar_cb_id,
        in_ntiles_c / in_nblocks_c,
        interm_reduction_cb_id,
        num_faces_in_tile,
        MAX_ROWS_FOR_REDUCTION);

    uint32_t interm_reduction_chunks = window_size_hw / MAX_ROWS_FOR_REDUCTION;
    cb_wait_front(in_scalar_cb_id, 1);
    //cb_wait_front(interm_reduction_cb_id, 1);
    cb_reserve_back(out_cb_id, 1);
    for (uint32_t i = 0; i < nsticks_per_core_by_nblocks; ++i) {
        for (uint32_t j = 0; j < num_8_tiles_blocks; j++) {
            // NOTE: Assuming in_ntiles_hw < 8 for now.
            // TODO: subblocking to support this.
            uint32_t out_write_idx = i * num_8_tiles_blocks + j;
            pack_untilize_dst_init_short<num_output_tiles>(
                interm_reduction_cb_id, num_out_rows, num_faces_in_tile);
            cb_reserve_back(interm_reduction_cb_id, 1);
            for (uint32_t h = 0; h <= interm_reduction_chunks; h++) {
                tile_regs_acquire();

                reduce_h_fused<in_ntiles_hw, in_ntiles_c, out_ntiles_c, in_nblocks_c, is_partial_tile, split_reader>(
                    in_cb_id,
                    in_scalar_cb_id,
                    i,
                    interm_reduction_cb_id,
                    MAX_ROWS_FOR_REDUCTION);
                tile_regs_wait();
                tile_regs_commit();
                pack_untilize_dst<num_output_tiles>(
                    interm_reduction_cb_id,
                    1 /*out_subblock_h*/,
                    h,
                    num_out_rows,
                    num_faces_in_tile); /* pack 1 row (1x16 or 1x32) */
                tile_regs_release();
            }
            cb_push_back(interm_reduction_cb_id, 1);
            pack_untilize_uninit(interm_reduction_cb_id);
            cb_wait_front(interm_reduction_cb_id, 1);

            pack_untilize_dst_init_short<num_output_tiles>(
                out_cb_id, num_out_rows, num_faces_in_tile);

            if(i == 0) {
                /*print_full_tile(interm_reduction_cb_id);*/
            }
            tile_regs_acquire();
            unpack_tilizeA_B_block(
                interm_reduction_cb_id,
                in_scalar_cb_id,
                in_ntiles_c / in_nblocks_c,
                0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
                num_faces_in_tile /* unpack 1 or 2 faces ) */,
                MAX_ROWS_FOR_REDUCTION);
            for (uint32_t c_i = 0; c_i < in_ntiles_c / in_nblocks_c; ++c_i) {
                reduce_tile_math(c_i, num_faces_in_tile /* reduce 1 or 2 faces */);
            }

            /*print_full_tile(out_cb_id);*/
            tile_regs_wait();
            tile_regs_commit();

            //print_tile_rows(out_cb_id, 1);

            pack_untilize_dst<num_output_tiles>(
                out_cb_id,
                1 /*out_subblock_h*/,
                out_write_idx,
                num_out_rows,
                num_faces_in_tile); /* pack 1 row (1x16 or 1x32) */
            tile_regs_release();
            cb_pop_front(interm_reduction_cb_id, 1);
            pack_untilize_uninit(out_cb_id);
        }
    }
    /*print_full_tile(out_cb_id);*/
    cb_push_back(out_cb_id, 1);
    cb_pop_front(in_scalar_cb_id, 1);
    //cb_pop_front(interm_reduction_cb_id, 1);
}

}  // namespace NAMESPACE
