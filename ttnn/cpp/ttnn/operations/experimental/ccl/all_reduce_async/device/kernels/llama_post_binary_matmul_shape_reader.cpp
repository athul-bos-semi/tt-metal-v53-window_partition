// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This should ideally be merged with `ccl_send_reader` when we are able to support compile time args
//       that don't require macros to function

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/buffer_constants.hpp>
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"

#include "cpp/ttnn/operations/ccl/common/kernels/command_processor.hpp"

#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"

#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/io_descriptors.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "cpp/ttnn/tensor/enum_types.hpp"
#include <cstdint>
#include <utility>

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb0_id = get_compile_time_arg_val(1);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(2);

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    // print every compile and runtime arg in uint32_t
    DPRINT << "ct args: \n";
    DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "num_tiles_per_core: " << (uint32_t)num_tiles_per_core << "\n";
    DPRINT << "num_tiles_to_read: " << (uint32_t)num_tiles_to_read << "\n";
    DPRINT << "first_core_tile_start_offset: " << (uint32_t)first_core_tile_start_offset << "\n";
    DPRINT << "num_cores: " << (uint32_t)num_cores << "\n";
    for (uint32_t i = 0; i < num_cores; i++) {
        DPRINT << "core_noc_x[" << i << "]: " << (uint32_t)core_noc_x[i] << "\n";
        DPRINT << "core_noc_y[" << i << "]: " << (uint32_t)core_noc_y[i] << "\n";
    }

    // interleaved addrgen

    DPRINT << "tensor -> CB: " << (uint32_t)cb0_id << "\n";

    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    while (tiles_read < num_tiles_to_read) {
        DPRINT << "tiles_read: " << tiles_read << "\n";
        uint32_t num_tiles_to_read_this_core =
            std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
        cb_reserve_back(cb0_id, num_tiles_to_read_this_core);
        const uint32_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t read_addr = get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0);
        read_addr += shard_tile_id * tensor0_page_size;

        noc_async_read(read_addr, l1_write_addr, num_tiles_to_read_this_core * tensor0_page_size);
        noc_async_read_barrier();

        cb_push_back(cb0_id, num_tiles_to_read_this_core);
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id = 0;
        core_id++;
    }

    DPRINT << "DONE \n";
}
