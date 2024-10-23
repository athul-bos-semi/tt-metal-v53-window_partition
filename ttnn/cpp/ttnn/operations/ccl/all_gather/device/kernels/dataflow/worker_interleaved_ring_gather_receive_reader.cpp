// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "risc_common.h"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp"


void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    volatile uint32_t *receiver_read_sem_addr = reinterpret_cast<volatile uint32_t *>(get_semaphore(get_compile_time_arg_val(1)));
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(3);
    constexpr ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode = static_cast<ttnn::ccl::EriscDataMoverTerminationMode>(get_compile_time_arg_val(4));

    uint32_t arg_idx = 0;
    const uint32_t eth_receiver_l1_base_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_transfers = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_full_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_pages_per_full_chunk = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rem_num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_receiver_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_receiver_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_receiver_l1_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);

    ASSERT(half_cb_n_pages > rem_num_pages);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    ccl::edm::WorkerToEdmReader<edm_termination_mode> reader(
        ttnn::ccl::WorkerXY(eth_receiver_noc_x, eth_receiver_noc_y),
        eth_receiver_l1_base_addr,
        num_buffers_per_channel,
        eth_receiver_l1_semaphore_addr,
        (num_full_chunks > 0 ? num_pages_per_full_chunk : rem_num_pages) * page_size,
        receiver_read_sem_addr);

    uint32_t myyx = (my_y[0] << 16) | my_x[0];
    bool last_message = false;
    DPRINT << "kernel" << myyx << "\n";
    for (uint32_t i = 0; i < num_transfers; ++i) {
        if (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                reader.wait_for_payload_available();
                if constexpr (edm_termination_mode == ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED) {
                    last_message = (i == num_transfers - 1 && c == num_full_chunks - 1 && rem_num_pages == 0);
                }
                if (last_message) {
                    DPRINT << "fetch_payload_blocking last_message" << myyx << "\n";
                } else {
                    DPRINT << "fetch_payload_blocking" << myyx << "\n";
                }
                reader.fetch_payload_blocking(cb_id_in0, num_pages_per_full_chunk, page_size, last_message);
            }
        }
        if (rem_num_pages > 0) {
            if constexpr (edm_termination_mode == ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED) {
                last_message = (i == num_transfers - 1);
            }
            reader.wait_for_payload_available();
                if (last_message) {
                    DPRINT << "fetch_payload_blocking last_message " << myyx << "\n";
                } else {
                    DPRINT << "fetch_payload_blocking" << myyx << "\n";
                }
            reader.fetch_payload_blocking(cb_id_in0, rem_num_pages, page_size, last_message);
            ASSERT(num_pages_per_full_chunk == 0 || num_pages_per_full_chunk > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
        }
    }

    if (num_transfers > 0 && (num_full_chunks > 0 || rem_num_pages > 0)) {
        DPRINT << "kernel" << myyx << " close\n";
        reader.close();
    }
    DPRINT << "kernel" << myyx << " DONE\n";
}
