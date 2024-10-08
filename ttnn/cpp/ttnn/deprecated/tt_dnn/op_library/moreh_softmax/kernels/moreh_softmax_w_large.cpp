// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_mask = tt::CB::c_in1;
    constexpr auto cb_bcast_scaler = tt::CB::c_in2;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_exps = tt::CB::c_intermed0;
    constexpr auto cb_recipsumexps = tt::CB::c_intermed1;
    constexpr auto cb_add = tt::CB::c_intermed2;
    constexpr auto cb_max = tt::CB::c_intermed3;
    constexpr auto cb_tmp = tt::CB::c_intermed4;
    constexpr auto cb_x_m_max = tt::CB::c_intermed5;

    binary_op_init_common(cb_in0, cb_bcast_scaler);

    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {

        // find max
        if (Wt == 1) {
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/1, /*popm=*/0);

            reduce_tile_to_cb<false, PoolType::MAX, REDUCE_DIM>(
                cb_tmp, cb_bcast_scaler, cb_max, Wt, /*pop0=*/1, /*pop1=*/0);
        } else {
            cb_reserve_back(cb_max, onetile);

            tile_regs_acquire();
            reduce_init_delta_with_dt<false, PoolType::MAX, REDUCE_DIM>(cb_max, cb_in0, cb_bcast_scaler);
            for (uint32_t w = 0; w < Wt - 1; ++w) {
                cb_wait_front(cb_in0, onetile);

                constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
                reduce_tile<PoolType::MAX, REDUCE_DIM>(cb_in0, cb_bcast_scaler, 0, bcast_scaler0, dst0);

                cb_pop_front(cb_in0, onetile);
            }
            reduce_revert_delta(cb_max);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_max);
            tile_regs_release();

            cb_push_back(cb_max, onetile);


            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/1, /*popm=*/0);


            cb_wait_front(cb_max, onetile);
            cb_wait_front(cb_tmp, onetile);

            tile_regs_acquire();
            copy_tile_init_with_dt(cb_max);
            copy_tile(cb_max, 0, dst0);

            constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
            reduce_init_delta_with_dt<false, PoolType::MAX, REDUCE_DIM>(cb_max, cb_tmp, cb_bcast_scaler);
            reduce_tile<PoolType::MAX, REDUCE_DIM>(cb_tmp, cb_bcast_scaler, 0, bcast_scaler0, dst0);
            reduce_revert_delta(cb_max);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_max);
            tile_regs_release();

            cb_pop_front(cb_max, onetile);
            cb_pop_front(cb_tmp, onetile);
            cb_push_back(cb_max, onetile);
        }

        // step 1
        for (uint32_t w = 0; w < Wt; ++w) {
            // compute exp(x)
            if (w == Wt - 1) {
                #ifdef SOFTMAX
                    sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

                    exp_tile_and_mask_tile_to_cb(
                        cb_tmp,
                        cb_mask,
                        cb_exps,
                        /*itile=*/0,
                        /*mtile=*/0,
                        /*pop=*/1,
                        /*popm=*/0);
                #else
                    rexp_tile_and_mask_tile_to_cb(
                        cb_in0,
                        cb_mask,
                        cb_exps,
                        /*itile=*/0,
                        /*mtile=*/0,
                        /*pop=*/1,
                        /*popm=*/0);
                #endif
            } else {
                #ifdef SOFTMAX
                    sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_x_m_max, 0, 0, /*pop0=*/1, /*pop1=*/0);
                    exp_tile_to_cb(cb_x_m_max, cb_exps);
                #else
                    sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_x_m_max, 0, 0, /*pop0=*/1, /*pop1=*/0);
                    rexp_tile_to_cb(cb_x_m_max, cb_exps);
                #endif
            }

            if (w == 0) {
                copy_tile_to_cb(cb_exps, cb_add);
            } else {
                cb_wait_front(cb_add, onetile);
                cb_wait_front(cb_exps, onetile);

                tile_regs_acquire();
                copy_tile_init_with_dt(cb_add);
                copy_tile(cb_add, 0, dst0);
                copy_tile_init_with_dt(cb_exps);
                copy_tile(cb_exps, 0, dst1);
                moreh_binary_op_init();
                moreh_binary_add(dst0);
                tile_regs_commit();

                cb_pop_front(cb_add, onetile);
                cb_pop_front(cb_exps, onetile);
                cb_reserve_back(cb_add, onetile);

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_add);
                tile_regs_release();

                cb_push_back(cb_add, onetile);
            }
        }

#ifdef LOG
        // compute log(sum)
        reduce_and_log_tile_to_cb<false, PoolType::SUM, REDUCE_DIM>(
            cb_add, cb_bcast_scaler, cb_recipsumexps, /*size=*/1, /*pop0=*/1, /*pop1=*/0);
#else
        // compute 1/sum(exp(x))
        reduce_and_recip_tile_to_cb<false, PoolType::SUM, REDUCE_DIM>(
             cb_add, cb_bcast_scaler, cb_recipsumexps, /*size=*/1, /*pop0=*/1, /*pop1=*/0);
#endif

        // step 3, compute final result
        for (uint32_t w = 0; w < Wt; w += onetile) {
            sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_x_m_max, 0, 0, /*pop0=*/1, /*pop1=*/0);
            cb_wait_front(cb_recipsumexps, onetile);
            cb_wait_front(cb_x_m_max, onetile);
            #ifdef LOG
                #ifdef SOFTMAX
                    // x - max - log(sum)
                    tile_regs_acquire();
                    copy_tile_init_with_dt(cb_x_m_max);
                    copy_tile(cb_x_m_max, 0, dst0);
                    copy_tile_init_with_dt(cb_recipsumexps);
                    copy_tile(cb_recipsumexps, 0, dst1);
                    moreh_binary_op_init();
                    moreh_binary_sub(dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_out0);
                    tile_regs_release();
                #else
                    // -x + max - log(sum)
                    // logsoftmin not implemented
                #endif
            #else
                #ifdef SOFTMAX
                    // exp(x - max) / sum
                    exp_tile_to_cb(cb_x_m_max, cb_exps);

                    tile_regs_acquire();
                    copy_tile_init_with_dt(cb_exps);
                    copy_tile(cb_exps, 0, dst0);
                    copy_tile_init_with_dt(cb_recipsumexps);
                    copy_tile(cb_recipsumexps, 0, dst1);
                    moreh_binary_op_init();
                    moreh_binary_mul(dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_out0);
                    tile_regs_release();

                    cb_pop_front(cb_exps, onetile);
                #else
                    // rexp(x - max) / sum
                    rexp_tile_to_cb(cb_x_m_max, cb_exps);

                    tile_regs_acquire();
                    copy_tile_init_with_dt(cb_exps);
                    copy_tile(cb_exps, 0, dst0);
                    copy_tile_init_with_dt(cb_recipsumexps);
                    copy_tile(cb_recipsumexps, 0, dst1);
                    moreh_binary_op_init();
                    moreh_binary_mul(dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_out0);
                    tile_regs_release();

                    cb_pop_front(cb_exps, onetile);
                #endif
            #endif
            cb_pop_front(cb_x_m_max, onetile);
            cb_push_back(cb_out0, onetile);
        }

        cb_pop_front(cb_recipsumexps, onetile);
        cb_pop_front(cb_max, onetile);
    }
}
}  // namespace NAMESPACE
