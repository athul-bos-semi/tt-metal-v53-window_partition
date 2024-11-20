// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

// clang-format off
#include "llrt/tt_cluster.hpp"
#include "umd/device/tt_xy_pair.h"
#include "llrt/tt_memory.h"
// clang-format on

// FIXME: ARCH_NAME specific include
#include "dev_mem_map.h" // MEM_LOCAL_BASE

namespace tt {

// llrt = lower-level runtime
namespace llrt {

using RamSrcAddr = unsigned int;
using RamDstAddr = unsigned int;
using SrcL1Core = CoreCoord;
using SrcL1Cores = std::vector<SrcL1Core>;
using DstL1Core = CoreCoord;
using DstL1Cores = std::vector<DstL1Core>;
using SrcChannelId = int;
using DstChannelId = int;
using DramBufferSize = unsigned int;
using DramSrcAddr = unsigned int;
using DramDstAddr = unsigned int;
using L1Addr = std::uint32_t;
using SrcAddr = std::uint32_t;
using DestAddr = std::uint32_t;
using LoadFirmwareFlag = bool;
using CountOffset = unsigned int;
using NCHW = std::array<std::uint32_t, 4>;
using RSUV = std::array<std::uint32_t, 4>;
using BYTES_PER_DATUM = std::uint32_t;
using TRANSACTION_SIZE = std::uint32_t;
using NUM_TRANSACTIONS = std::uint32_t;
using NUM_REPETITIONS = std::uint32_t;

using WorkerCore = tt_cxy_pair;
using WorkerCores = std::vector<WorkerCore>;

// Return a reference to a potentially shared binary image.
// The images are cached by path name, which is never erased.
// TODO: Remove core_type_idx, processor_class_idx,
// processor_type_idx -- the information they provide can be
// obtained directly from the binary image.
ll_api::memory const& get_risc_binary(
    string const& path,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx,
    ll_api::memory::PackSpans span_type = ll_api::memory::PackSpans::NO_PACK,
    ll_api::memory::Relocate relo_type = ll_api::memory::Relocate::NONE);

// TODO: try using "stop" method from device instead, it's the proper way of asserting reset

// CoreCoord core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor
template<typename DType>
void write_hex_vec_to_core(
    chip_id_t chip,
    const CoreCoord &core,
    const std::vector<DType>& hex_vec,
    uint64_t addr,
    bool small_access = false) {
    tt::Cluster::instance().write_core(hex_vec.data(), hex_vec.size() * sizeof(DType), tt_cxy_pair(chip, core), addr, small_access);
}
template<typename DType>
void write_hex_vec_to_core(
    chip_id_t chip,
    const CoreCoord &core,
    tt::stl::Span<const DType> hex_vec,
    uint64_t addr,
    bool small_access = false) {
    tt::Cluster::instance().write_core(hex_vec.data(), hex_vec.size() * sizeof(DType), tt_cxy_pair(chip, core), addr, small_access);
}

std::vector<std::uint32_t> read_hex_vec_from_core(chip_id_t chip, const CoreCoord &core, uint64_t addr, uint32_t size);

CoreCoord logical_core_from_ethernet_core(chip_id_t chip_id, CoreCoord &physical_core);

void write_launch_msg_to_core(chip_id_t chip, CoreCoord core, launch_msg_t *msg, go_msg_t * go_msg, uint64_t addr, bool send_go = true);

void print_worker_cores(chip_id_t chip_id = 0);

inline bool is_worker_core(const CoreCoord &core, chip_id_t chip_id) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(chip_id);
    return std::find(soc_desc.physical_workers.begin(), soc_desc.physical_workers.end(), core) !=
           soc_desc.physical_workers.end();
}

inline bool is_ethernet_core(const CoreCoord &core, chip_id_t chip_id) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(chip_id);
    return std::find(soc_desc.physical_ethernet_cores.begin(), soc_desc.physical_ethernet_cores.end(), core) !=
           soc_desc.physical_ethernet_cores.end();
}

bool test_load_write_read_risc_binary(
    ll_api::memory const& mem,
    chip_id_t chip_id,
    const CoreCoord& core,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx);
void write_binary_to_address(ll_api::memory const& mem, chip_id_t chip_id, const CoreCoord& core, uint32_t address);

// subchannel hard-coded to 0 for now
CoreCoord get_core_for_dram_channel(int dram_channel_id, chip_id_t chip_id = 0);

namespace internal_ {

void wait_until_cores_done(
    chip_id_t device_id, int run_state, std::unordered_set<CoreCoord> &not_done_phys_cores, int timeout_ms = 0);

}  // namespace internal_

}  // namespace llrt

}  // namespace tt
