// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint_pages.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    DPRINT << "alooo" << ENDL();
    uint32_t l1_read_addr = get_read_ptr(cb_id_in0);
    tt::data_movement::common::print_bf16_pages(l1_read_addr, 8, 32, 0);

    cb_push_back(cb_id_in0, num_tiles_per_core);
}
