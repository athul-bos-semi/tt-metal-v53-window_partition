// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    unary_bcast_init<BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_wait_front(tt::CBIndex::c_0, per_core_block_dim);
        acquire_dst();
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            unary_bcast<BCAST_DIM>(tt::CBIndex::c_0, tile_index, tile_index);
        }

        cb_pop_front(tt::CBIndex::c_0, per_core_block_dim);
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            pack_tile(tile_index, tt::CBIndex::c_16);
        }

        cb_push_back(tt::CBIndex::c_16, per_core_block_dim);
        release_dst();
    }
}
}  // namespace NAMESPACE
