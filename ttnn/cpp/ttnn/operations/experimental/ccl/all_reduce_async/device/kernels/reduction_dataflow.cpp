// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t total_num_reduction_tiles = get_compile_time_arg_val(1);
    const uint32_t signal_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    // 1. Wait for signal from All-Gather worker
    noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
    noc_semaphore_set(signal_semaphore_addr_ptr, 0);

    // 2. Signal compute kernel to start processing
    cb_push_back(cb_id, total_num_reduction_tiles);
}
