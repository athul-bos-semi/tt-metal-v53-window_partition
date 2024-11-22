// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL


#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"


ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }


namespace NAMESPACE {
void MAIN {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t do_beta = get_compile_time_arg_val(3);


    #ifdef FUSE_PRE_ADD
        binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1);
    #else
        binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0);
    #endif

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // TODO(AP): check that if DST is indeed zeroed by release_dst (and initially), we can use it as zeroes

    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_scaler = tt::CBIndex::c_2; // single tile generated by the reader
    constexpr auto cb_eps = tt::CBIndex::c_3; // single tile generated by the reader
    constexpr auto cb_xmm = tt::CBIndex::c_24; // x minus mean, this is a large buffer, see setup code in layernorm_op.cpp
    constexpr auto cb_ex = tt::CBIndex::c_25; // E[x]
    constexpr auto cb_ex2 = tt::CBIndex::c_26; // E[(x-E[x])^2]
    constexpr auto cb_xmm2 = tt::CBIndex::c_27; // xmm^2
    constexpr auto cb_ex2pe = tt::CBIndex::c_28; // E[(x-E[x])^2]+eps
    constexpr auto cb_in = tt::CBIndex::c_0; // input x or a for fused pre-add (x=a+b)
    constexpr auto cb_inb = tt::CBIndex::c_1; // input b for fused pre-add
    constexpr auto cb_out = tt::CBIndex::c_16; // output
    constexpr auto cb_gamma = tt::CBIndex::c_5;
    constexpr auto cb_beta = tt::CBIndex::c_6;
    constexpr auto cb_fusion = tt::CBIndex::c_29; // stream gamma/beta
    constexpr auto scaler0 = 0;
    #ifdef FUSE_PRE_ADD
    constexpr auto cb_x = tt::CBIndex::c_30;
    #else
    constexpr auto cb_x = tt::CBIndex::c_0;
    #endif

    cb_wait_front(cb_scaler, 1); // comes from the reader
    cb_wait_front(cb_eps, 1); // comes from the reader


    constexpr int cb_im_or_out = (do_gamma|do_beta) ? cb_fusion : tt::CBIndex::c_16;


    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {

        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        /*
         * X + Y
         */
        #ifdef FUSE_PRE_ADD
            add_tiles_init();
            for (uint32_t wt = 0; wt < Wt; wt += blk) {
                ACQ();
                        //UNPACK(( { DPRINT  << "Waiting on cb_x" << ENDL(); } ));
                cb_wait_front(cb_in, blk);
                        //UNPACK(( { DPRINT  << "Waiting on cb_inb" << ENDL(); } ));
                cb_wait_front(cb_inb, blk);
                        //UNPACK(( { DPRINT  << "Done Waiting on cb_inb" << ENDL(); } ));
                cb_reserve_back(cb_x, blk);
                for (uint32_t j = 0; j < blk; j++) {
                    add_tiles(cb_in, cb_inb, j, j, j);
                    pack_tile(j, cb_x);
                }
                REL();
                cb_push_back(cb_x, blk); // push the sum into the same buffer
                cb_pop_front(cb_in, blk);
                cb_pop_front(cb_inb, blk);
            }
            // by the end of this loop we should end up with Wt tiles in cb_x
        #endif

        /*
         * E[x]
         * means = tensor.sum(x, 3, scalar=1.0/W) # -> NCH1
         */
        ACQ();
        cb_reserve_back(cb_ex, 1*onetile);
        reduce_init_delta<false>();
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_x, wt+blk);
            for (uint32_t j = 0; j < blk; j++) {
                reduce_tile(cb_x, cb_scaler, wt+j, scaler0, dst0);
            }
            // we don't pop cb_x until we compute Ex
        }
        pack_tile(dst0, cb_ex);
        reduce_revert_delta();
        REL();

        cb_push_back(cb_ex, 1);

        /*
         * x - E[x]
         * compute xmm=x-mean. Reuse cb_x since we didn't pop anything from it
         */
        cb_wait_front(cb_ex, 1); // should have 1 tile
        cb_reserve_back(cb_xmm, Wt);
        sub_bcast_cols_init_short();
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            ACQ();
            for (uint32_t wtr = 0; wtr<blk; wtr++) {
                sub_tiles_bcast_cols(cb_x, cb_ex, wt+wtr, 0, wtr); // tile *= 1/(sum(exp(x)))
                pack_tile(wtr, cb_xmm);
            }
            cb_push_back(cb_xmm, blk);
            REL();
        }
        cb_pop_front(cb_ex, 1);
        cb_pop_front(cb_x, Wt);

        /* (x - E[x])^2
         * compute temp = xmm*xmm = (x-E[x])^2
         */
        mul_tiles_init();
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_xmm, wt+blk); // cumulative wait
            cb_reserve_back(cb_xmm2, blk); // can probably use less space for this if we block
            ACQ();
            for (uint32_t wtr = 0; wtr<blk; wtr++) {
                mul_tiles(cb_xmm, cb_xmm, wt+wtr, wt+wtr, wtr);
                //mul_tiles(cb_xmm, cb_col1, wt+wtr, wt+wtr, wtr);
                pack_tile(wtr, cb_xmm2);
            }
            cb_push_back(cb_xmm2, blk);
            REL();
        }

        /* Var(x)
         * compute E[(x-E[x])^2]
         * IIRC E[x^2] - E[x]^2 trick was unstable
         * TODO(AP): can save space here by reusing CB
         */
        cb_reserve_back(cb_ex2, 1);
        reduce_init_delta<false>();
        ACQ();
        cb_wait_front(cb_xmm2, Wt);
        //cb_wait_front(cb_xmm, Wt);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            // reduce
            for (uint32_t wtr = 0; wtr<blk; wtr++)
                reduce_tile(cb_xmm2, cb_scaler, wt+wtr, scaler0, dst0);
                //reduce_tile(cb_xmm, cb_scaler, wt+wtr, scaler0, dst0);
        }
        cb_pop_front(cb_xmm2, Wt);
        pack_tile(dst0, cb_ex2);
        reduce_revert_delta();
        REL();

        cb_push_back(cb_ex2, 1);
        cb_wait_front(cb_ex2, 1);

        /* Var(x) + eps
         * add epsilon E[(x-E[x])^2]+eps
         */
        ACQ();
        add_tiles_init();
        add_tiles(cb_ex2, cb_eps, 0, 0, dst0);

        cb_reserve_back(cb_ex2pe, 1); // 1
        sqrt_tile_init();
        sqrt_tile(dst0);
        recip_tile_init();
        recip_tile(dst0);
        pack_tile(dst0, cb_ex2pe);
        cb_push_back(cb_ex2pe, 1);
        REL();
        cb_pop_front(cb_ex2, 1);

        /* ln(x) * gamma + beta (gamma and beta are optional)
         * now xmm = (x-E[x])
         * we have 1.0/sqrt( E[(x-E[x])^2] + eps) in cb_ex2pe
         * just need to bcast_mul xmm with cb_ex2pe
         */
        cb_wait_front(cb_ex2pe, 1);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
                        //if (ht == 1) UNPACK(( DPRINT << "wt_2=" << wt << " " ));
                        //if (ht == 1) UNPACK(( DPRINT << "rem_2=" << rem << ENDL() ));
            cb_reserve_back(cb_im_or_out, blk);

            ACQ();
            mul_bcast_cols_init_short();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                // cb_xmm[wt+wtr] since we pop Wt from cb_xmm after the entire loop
                mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, wt+wtr, 0, wtr); // tile *= 1/(sum(exp(x)))
                pack_tile(wtr, cb_im_or_out); // pack either to intermediate (cb_fusion or out0)
            }
            cb_push_back(cb_im_or_out, blk); // if no gamma/beta are provided, this will be passed on to the writer
            REL();

            if (do_gamma) {
                ACQ();
                uint32_t cb_outg = do_beta ? cb_fusion : tt::CBIndex::c_16;
                mul_bcast_rows_init_short();
                cb_reserve_back(cb_outg, blk);
                cb_wait_front(cb_gamma, wt+blk); // we don't pop, TODO: only wait on first ht
                cb_wait_front(cb_fusion, blk);
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    mul_tiles_bcast_rows(cb_fusion, cb_gamma, wtr, wt+wtr, wtr); // tile *= 1/(sum(exp(x)))
                    pack_tile(wtr, cb_outg); // pack either to intermediate (cb_fusion or out0)
                }
                cb_pop_front(cb_fusion, blk);
                // we don't pop gamma
                cb_push_back(cb_outg, blk);
                // We don't pop gamma since it's 1,1,1,Wt and we reuse it for all NCHt
                REL();
            }
            if (do_beta) {
                ACQ();
                add_bcast_rows_init_short();
                cb_reserve_back(tt::CBIndex::c_16, blk);
                cb_wait_front(cb_beta, wt+blk); // TODO: optimization - only wait on first ht
                cb_wait_front(cb_fusion, blk);
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    add_tiles_bcast_rows(cb_fusion, cb_beta, wtr, wt+wtr, wtr); // tile *= 1/(sum(exp(x)))
                    pack_tile(wtr, tt::CBIndex::c_16); // pack either to intermediate (cb_fusion or out0)
                }
                cb_pop_front(cb_fusion, blk);
                // We don't pop beta since it's 1,1,1,Wt and we reuse it for all NCHt
                cb_push_back(tt::CBIndex::c_16, blk);
                REL();
            }
        }
        cb_pop_front(cb_ex2pe, 1);
        cb_pop_front(cb_xmm, Wt);

    } // NCHt loop
    //cb_pop_front(cb_scaler, 1); // optional for correctness
    //cb_pop_front(cb_eps, 1); // optional for correctness
    //cb_pop_front(cb_col1, 1); // optional for correctness
}
}
