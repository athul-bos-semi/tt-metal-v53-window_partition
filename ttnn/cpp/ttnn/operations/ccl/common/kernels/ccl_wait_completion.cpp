// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "dataflow_api.h"

#include <cstdint>
#include <array>

#include "debug/dprint.h"

void kernel_main() {
    constexpr size_t num_signals_to_wait_for = get_compile_time_arg_val(0);
    constexpr size_t send_termination_signals = get_compile_time_arg_val(1);
    std::array<volatile uint32_t*, num_signals_to_wait_for> sem_addrs;
    std::array<size_t, num_signals_to_wait_for> expected_sem_counts;
    std::array<size_t, num_signals_to_wait_for> current_sem_counts;


    size_t arg_idx = 0;
    for (size_t i = 0; i < num_signals_to_wait_for; ++i) {
        sem_addrs[i] = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
        DPRINT << "DRAIN WAITING ON SEMAPHORE ADDR " << (uint32_t)sem_addrs[i] << " on core " << (uint32_t)((my_y[0] << 16) | my_x[0]) << "\n";
        expected_sem_counts[i] = get_arg_val<uint32_t>(arg_idx++);
        current_sem_counts[i] = 0;
    }

    while (true) {
        for (size_t i = 0; i < num_signals_to_wait_for; ++i) {
            if (current_sem_counts[i] >= expected_sem_counts[i]) {
                continue;
            }

            ASSERT(noc_x == my_x[0] && noc_y == my_y[0]);// "semaphore address is not a local semaphore. Only local semaphore address is support at this time"
            if (current_sem_counts[i] != *sem_addrs[i]) {
                DPRINT << "DRAIN GOT SEMINC @ " << (uint32_t)sem_addrs[i] << ". NOW= " << (uint32_t)*sem_addrs[i] << "\n";
                current_sem_counts[i] = *sem_addrs[i];
            }
        }

        bool all_done = true;
        for (size_t i = 0; i < num_signals_to_wait_for; ++i) {
            if (current_sem_counts[i] < expected_sem_counts[i]) {
                all_done = false;
                break;
            }
        }
        if (all_done) {
            break;
        }
    }

    DPRINT << "DONE RECEIVING SEMINCS. SHUTTING DOWN FABRIC\n";

    if (send_termination_signals) {
        size_t num_termination_signals = get_arg_val<uint32_t>(arg_idx++);
        for (size_t i = 0; i < num_termination_signals; ++i) {
            uint64_t termination_addr = get_arg_val<uint32_t>(arg_idx++);
            uint32_t noc_x = get_arg_val<uint32_t>(arg_idx++);
            uint32_t noc_y = get_arg_val<uint32_t>(arg_idx++);
            uint32_t addr = get_arg_val<uint32_t>(arg_idx++);
            noc_semaphore_inc(get_noc_addr(noc_x, noc_y, addr), 1);
        }
    }
    DPRINT << "DRAIN DONE\n";
}
