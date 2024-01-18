// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_norm/kernel_utils/common_ckernels.hpp"

namespace NAMESPACE {
void MAIN {
    int i{0};
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto num_reduced_tiles_along_dim = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto recip_p = get_arg_val<uint32_t>(i++);
    const bool recip_p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    std::uint8_t input_id{0};
    const auto cb_x = input_id++;                // input
    const auto cb_one = input_id++;              // one
    const auto cb_decimal = input_id++;          // decimal
    const auto cb_recip_p_decimal = input_id++;  // recip_p_decimal

    std::uint8_t output_id{16};
    const auto cb_y = output_id++;  // output

    std::uint8_t intermed_id{24};
    const auto cb_tmp0 = intermed_id++;
    const auto cb_tmp1 = intermed_id++;
    const auto cb_tmp2 = intermed_id++;
    const auto cb_tmp3 = intermed_id++;
    const auto cb_tmp4 = intermed_id++;
    const auto cb_tmp5 = intermed_id++;

    const auto cb_xabs = cb_tmp0;          // |x|
    const auto cb_xpow = cb_tmp1;          // |x|^p
    const auto cb_logx = cb_tmp2;          // log(|x|)
    const auto cb_exp_lxmd = cb_tmp3;      // exp(log(|x|) * decimal)
    const auto cb_correct_xpow = cb_tmp4;  // |x|^p * exp(log(|x|) * decimal)(==|x + decimal|^p)
    const auto cb_xpowadd = cb_tmp5;       // Add(|x + decimal|^p)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);

    cb_wait_front(cb_one, onetile);              // comes from the reader
    cb_wait_front(cb_decimal, onetile);          // comes from the reader
    cb_wait_front(cb_recip_p_decimal, onetile);  // comes from the reader

    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            // |x|
            ACQ();
            cb_wait_front(cb_x, onetile);  // comes from the reader
            cb_reserve_back(cb_xabs, onetile);

            copy_tile_init();
            copy_tile(cb_x, 0, dst0);

            abs_tile_init();
            abs_tile(dst0);

            pack_tile(dst0, cb_xabs);

            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_xabs, onetile);
            REL();

            power_tile_to_cb(cb_xabs, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_correct_xpow, p, p_is_negative);

            // Add(|x|^p)
            if (inner_idx == 0) {
                ACQ();
                cb_wait_front(cb_correct_xpow, onetile);
                cb_reserve_back(cb_xpowadd, onetile);

                copy_tile_init();
                copy_tile(cb_correct_xpow, 0, dst0);

                pack_tile(dst0, cb_xpowadd);

                cb_pop_front(cb_correct_xpow, onetile);
                cb_push_back(cb_xpowadd, onetile);
                REL();
            } else {
                ACQ();
                cb_wait_front(cb_correct_xpow, onetile);
                cb_wait_front(cb_xpowadd, onetile);
                cb_reserve_back(cb_xpowadd, onetile);

                add_tiles_init();
                add_tiles(cb_correct_xpow, cb_xpowadd, 0, 0, dst0);

                pack_tile(dst0, cb_xpowadd);

                cb_pop_front(cb_correct_xpow, onetile);
                cb_pop_front(cb_xpowadd, onetile);
                cb_push_back(cb_xpowadd, onetile);
                REL();
            }
        }

        // Compute cb_y
        power_tile_to_cb(cb_xpowadd, cb_tmp0, cb_tmp1, cb_recip_p_decimal, cb_tmp2, cb_y, recip_p, recip_p_is_negative);
    }
    cb_pop_front(cb_one, onetile);
    cb_pop_front(cb_decimal, onetile);
    cb_pop_front(cb_recip_p_decimal, onetile);

}  // void MAIN
}  // namespace NAMESPACE
