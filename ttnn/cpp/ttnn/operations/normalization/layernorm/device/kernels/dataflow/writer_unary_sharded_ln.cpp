// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

#include "debug/dprint.h"

void kernel_main() {
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta = get_compile_time_arg_val(2) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool beta_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t block_w = get_compile_time_arg_val(5);

    // Reshard writer
    constexpr uint32_t worker_core_stride_w_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t storage_core_stride_w_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t block_ht = get_compile_time_arg_val(12);

    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(5);
    const uint32_t beta_tile_start_id = get_arg_val<uint32_t>(6);

    // Reshard writer
    const uint32_t num_segments_to_write_back = get_arg_val<uint32_t>(7);
    const uint32_t storage_core_start_offset = get_arg_val<uint32_t>(8);
    tt_l1_ptr uint32_t* segment_args = (tt_l1_ptr uint32_t*)(get_arg_addr(9));

    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_resharded = tt::CBIndex::c_17;

    {
        constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
        const uint32_t scalar_w = get_arg_val<uint32_t>(1);
        generate_reduce_scaler(cb_in_2, scalar_w);
    }
    if constexpr (is_all_to_all_worker) {
        constexpr uint32_t cb_in_4 = tt::CBIndex::c_4;
        const uint32_t scalar_c = get_arg_val<uint32_t>(0);
        generate_reduce_scaler(cb_in_4, scalar_c);
    }
    constexpr uint32_t eps_cb_id = 3;
    const uint32_t eps = get_arg_val<uint32_t>(2);
    generate_bcast_col_scalar(eps_cb_id, eps);

    if constexpr (fuse_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        const DataFormat gamma_data_format = get_dataformat(cb_gamma);
        const InterleavedAddrGenFast<gamma_is_dram> gamma = {
            .bank_base_address = gamma_addr, .page_size = gamma_tile_bytes, .data_format = gamma_data_format};

        uint32_t l1_write_addr_gamma = get_write_ptr(cb_gamma);
        cb_reserve_back(cb_gamma, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            noc_async_read_tile(tile_id, gamma, l1_write_addr_gamma);
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, block_w);
    }

    if constexpr (fuse_beta) {
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
        const DataFormat beta_data_format = get_dataformat(cb_beta);
        const InterleavedAddrGenFast<beta_is_dram> beta = {
            .bank_base_address = beta_addr, .page_size = beta_tile_bytes, .data_format = beta_data_format};

        uint32_t l1_write_addr_beta = get_write_ptr(cb_beta);
        cb_reserve_back(cb_beta, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = beta_tile_start_id + w;
            noc_async_read_tile(tile_id, beta, l1_write_addr_beta);
            l1_write_addr_beta += beta_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta, block_w);
    }

#ifndef SKIP_WRITE_BACK

    uint32_t args_idx = 0;
    uint32_t worker_core_read_offset = 0;

    cb_wait_front(cb_out, block_ht * block_w);
    uint32_t cb_out_read_base_addr = get_read_ptr(cb_out);
    uint32_t cb_out_reshard_write_base_addr = get_write_ptr(cb_out_resharded);

    for (uint32_t i = 0; i < num_segments_to_write_back; ++i) {
        uint32_t write_size = segment_args[args_idx++];
        uint32_t storage_core_x = segment_args[args_idx++];
        uint32_t storage_core_y = segment_args[args_idx++];

        uint32_t worker_core_read_addr = cb_out_read_base_addr + worker_core_read_offset;
        uint32_t local_storage_core_write_addr = cb_out_reshard_write_base_addr;
        if (i == 0) {  // For the first segment we need to add the start offset; the following segments will start at 0
                       // offset
            local_storage_core_write_addr += storage_core_start_offset;
        }

        uint64_t remote_storage_core_write_addr =
            get_noc_addr(storage_core_x, storage_core_y, local_storage_core_write_addr);

        for (uint32_t h = 0; h < block_ht; ++h) {
            noc_async_write(worker_core_read_addr, remote_storage_core_write_addr, write_size);
            worker_core_read_addr += worker_core_stride_w_bytes;
            remote_storage_core_write_addr += storage_core_stride_w_bytes;
        }
        worker_core_read_offset += write_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, block_ht * block_w);
#endif
}
