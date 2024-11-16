// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// debug/sanitize_noc.h
//
// This file implements a method sanitize noc addresses.
// Malformed addresses (out of range offsets, bad XY, etc) are stored in L1
// where the watcher thread can log the result.  The device then soft-hangs in
// a spin loop.
//
// All functionaly gated behind defined WATCHER_ENABLED
//
#pragma once

// NOC logging enabled independently of watcher, need to include it here because it hooks into DEBUG_SANITIZE_NOC_*
#include "noc_logging.h"

#if (                                                                                          \
    defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC)) &&                                                        \
    defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_NOC_SANITIZE) && !defined(FORCE_WATCHER_OFF)

#include "watcher_common.h"

#include "debug/waypoint.h"

#include "dev_msgs.h"
#include "noc_overlay_parameters.h"
#include "noc_parameters.h"
#include "noc_nonblocking_api.h"

// A couple defines for specifying read/write and multi/unicast
#define DEBUG_SANITIZE_NOC_READ true
#define DEBUG_SANITIZE_NOC_WRITE false
typedef bool debug_sanitize_noc_dir_t;
#define DEBUG_SANITIZE_NOC_MULTICAST true
#define DEBUG_SANITIZE_NOC_UNICAST false
typedef bool debug_sanitize_noc_cast_t;
#define DEBUG_SANITIZE_NOC_TARGET true
#define DEBUG_SANITIZE_NOC_LOCAL false
typedef bool debug_sanitize_noc_which_core_t;

// Helper function to get the core type from noc coords.
AddressableCoreType get_core_type(uint8_t noc_id, uint8_t x, uint8_t y) {
    core_info_msg_t tt_l1_ptr *core_info = GET_MAILBOX_ADDRESS_DEV(core_info);

    for (uint32_t idx = 0; idx < MAX_NON_WORKER_CORES; idx++) {
        uint8_t core_x = core_info->non_worker_cores[idx].x;
        uint8_t core_y = core_info->non_worker_cores[idx].y;
        if (x == NOC_0_X(noc_id, core_info->noc_size_x, (uint32_t) core_x) &&
            y == NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) core_y)) {
            return core_info->non_worker_cores[idx].type;
        }
    }

    for (uint32_t idx = 0; idx < MAX_HARVESTED_ROWS; idx++) {
        uint16_t harvested_y = core_info->harvested_y[idx];
        if (y == NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) harvested_y)) {
            return AddressableCoreType::HARVESTED;
        }
    }

    // Tensix
    if (noc_id == 0) {
        if (x >= NOC_0_X(noc_id, core_info->noc_size_x, (uint32_t) 1) &&
            x <= NOC_0_X(noc_id, core_info->noc_size_x, (uint32_t) core_info->noc_size_x - 1) &&
            y >= NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) 1) &&
            y <= NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) core_info->noc_size_y - 1)) {
            return AddressableCoreType::TENSIX;
        }
    } else {
        if (x <= NOC_0_X(noc_id, core_info->noc_size_x, (uint32_t) 1) &&
            x >= NOC_0_X(noc_id, core_info->noc_size_x, (uint32_t) core_info->noc_size_x - 1) &&
            y <= NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) 1) &&
            y >= NOC_0_Y(noc_id, core_info->noc_size_y, (uint32_t) core_info->noc_size_y - 1)) {
            return AddressableCoreType::TENSIX;
        }
    }

    return AddressableCoreType::UNKNOWN;
}

// TODO(PGK): remove soft reset when fw is downloaded at init
inline bool debug_valid_reg_addr(uint64_t addr, uint64_t len) {
    return (((addr >= NOC_OVERLAY_START_ADDR) &&
             (addr < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) ||
            (addr == RISCV_DEBUG_REG_SOFT_RESET_0)) &&
           (len == 4);
}

inline uint16_t debug_valid_worker_addr(uint64_t addr, uint64_t len) {
    if (addr + len <= addr)
        return DebugSanitizeNocAddrZeroLength;
    if (addr < MEM_L1_BASE)
        return DebugSanitizeNocAddrUnderflow;
    if (addr + len > MEM_L1_BASE + MEM_L1_SIZE)
        return DebugSanitizeNocAddrOverflow;
    return DebugSanitizeNocOK;
}

inline uint16_t debug_valid_pcie_addr(uint64_t addr, uint64_t len) {
    if (addr + len <= addr)
        return DebugSanitizeNocAddrZeroLength;

    core_info_msg_t tt_l1_ptr *core_info = GET_MAILBOX_ADDRESS_DEV(core_info);
    if (addr < core_info->noc_pcie_addr_base)
        return DebugSanitizeNocAddrUnderflow;
    if (addr + len > core_info->noc_pcie_addr_end)
        return DebugSanitizeNocAddrOverflow;
    return DebugSanitizeNocOK;
}
inline uint16_t debug_valid_dram_addr(uint64_t addr, uint64_t len) {
    if (addr + len <= addr)
        return DebugSanitizeNocAddrZeroLength;

    core_info_msg_t tt_l1_ptr *core_info = GET_MAILBOX_ADDRESS_DEV(core_info);
    if (addr < core_info->noc_dram_addr_base)
        return DebugSanitizeNocAddrUnderflow;
    if (addr + len > core_info->noc_dram_addr_end)
        return DebugSanitizeNocAddrOverflow;
    return DebugSanitizeNocOK;
}

inline uint16_t debug_valid_eth_addr(uint64_t addr, uint64_t len) {
    if (addr + len <= addr)
        return DebugSanitizeNocAddrZeroLength;
    if (addr < MEM_ETH_BASE)
        return DebugSanitizeNocAddrUnderflow;
    if (addr + len > MEM_ETH_BASE + MEM_ETH_SIZE)
        return DebugSanitizeNocAddrOverflow;
    return DebugSanitizeNocOK;
}

// Note:
//  - this isn't racy w/ the host so long as invalid is written last
//  - this isn't racy between riscvs so long as each gets their own noc_index
inline void debug_sanitize_post_noc_addr_and_hang(
    uint8_t noc_id,
    uint64_t noc_addr,
    uint32_t l1_addr,
    uint32_t len,
    debug_sanitize_noc_cast_t multicast,
    debug_sanitize_noc_dir_t dir,
    debug_sanitize_noc_which_core_t which_core,
    uint16_t return_code) {
    if (return_code == DebugSanitizeNocOK)
        return;

    debug_sanitize_noc_addr_msg_t tt_l1_ptr *v = *GET_MAILBOX_ADDRESS_DEV(watcher.sanitize_noc);

    if (v[noc_id].return_code == DebugSanitizeNocOK) {
        v[noc_id].noc_addr = noc_addr;
        v[noc_id].l1_addr = l1_addr;
        v[noc_id].len = len;
        v[noc_id].which_risc = debug_get_which_riscv();
        v[noc_id].is_multicast = (multicast == DEBUG_SANITIZE_NOC_MULTICAST);
        v[noc_id].is_write = (dir == DEBUG_SANITIZE_NOC_WRITE);
        v[noc_id].is_target = (which_core == DEBUG_SANITIZE_NOC_TARGET);
        v[noc_id].return_code = return_code;
    }
    WAYPOINT("XXXX");

#if defined(COMPILE_FOR_ERISC)
    // Update launch msg to show that we've exited. This is required so that the next run doesn't think there's a kernel
    // still running and try to make it exit.
    tt_l1_ptr go_msg_t *go_message_ptr = GET_MAILBOX_ADDRESS_DEV(go_message);
    go_message_ptr->signal = RUN_MSG_DONE;

    // For erisc, we can't hang the kernel/fw, because the core doesn't get restarted when a new
    // kernel is written. In this case we'll do an early exit back to base FW.
   // internal_::disable_erisc_app();
   // erisc_exit();
#endif


    while (1) { ; }
}

// Return value is the alignment mask for the type of core the noc address points
// to. Need to do this because L1 alignment needs to match the noc address alignment requirements,
// even if it's different than the inherent L1 alignment requirements.
// Direction is specified because reads and writes may have different L1 requirements (see noc_parameters.h).
uint32_t debug_sanitize_noc_addr(
    uint8_t noc_id,
    uint64_t noc_addr,
    uint32_t l1_addr,
    uint32_t noc_len,
    debug_sanitize_noc_cast_t multicast,
    debug_sanitize_noc_dir_t dir) {
    // Different encoding of noc addr depending on multicast vs unitcast
    uint8_t x, y;
    if (multicast) {
        x = (uint8_t) NOC_MCAST_ADDR_START_X(noc_addr);
        y = (uint8_t) NOC_MCAST_ADDR_START_Y(noc_addr);
    } else {
        x = (uint8_t) NOC_UNICAST_ADDR_X(noc_addr);
        y = (uint8_t) NOC_UNICAST_ADDR_Y(noc_addr);
    }
    uint64_t noc_local_addr = NOC_LOCAL_ADDR(noc_addr);
    AddressableCoreType core_type = get_core_type(noc_id, x, y);

    // Extra check for multicast
    if (multicast) {
        uint8_t x_end = (uint8_t) NOC_MCAST_ADDR_END_X(noc_addr);
        uint8_t y_end = (uint8_t) NOC_MCAST_ADDR_END_Y(noc_addr);

        AddressableCoreType end_core_type = get_core_type(noc_id, x_end, y_end);

        // Multicast supports workers only
        uint16_t return_code = DebugSanitizeNocOK;
        if (core_type != AddressableCoreType::TENSIX || end_core_type != AddressableCoreType::TENSIX)
            return_code = DebugSanitizeNocMulticastNonWorker;
        if (x > x_end || y > y_end)
            return_code = DebugSanitizeNocMulticastInvalidRange;
        debug_sanitize_post_noc_addr_and_hang(
            noc_id, noc_addr, l1_addr, noc_len, multicast, dir, DEBUG_SANITIZE_NOC_TARGET, return_code);
    }

    // Check noc addr, we save the alignment requirement from the noc src/dst because the L1 address
    // needs to match alignment.
    // Reads and writes may have different alignment requirements, see noc_parameters.h for details.
    uint32_t alignment_mask = (dir == DEBUG_SANITIZE_NOC_READ ? NOC_L1_READ_ALIGNMENT_BYTES : NOC_L1_WRITE_ALIGNMENT_BYTES) - 1;  // Default alignment, only override in ceratin cases.
    if (core_type == AddressableCoreType::PCIE) {
        alignment_mask = (dir == DEBUG_SANITIZE_NOC_READ ? NOC_PCIE_READ_ALIGNMENT_BYTES : NOC_PCIE_WRITE_ALIGNMENT_BYTES) - 1;
        debug_sanitize_post_noc_addr_and_hang(
            noc_id, noc_addr, l1_addr, noc_len, multicast, dir, DEBUG_SANITIZE_NOC_TARGET, debug_valid_pcie_addr(noc_local_addr, noc_len));
    } else if (core_type == AddressableCoreType::DRAM) {
        alignment_mask = (dir == DEBUG_SANITIZE_NOC_READ ? NOC_DRAM_READ_ALIGNMENT_BYTES : NOC_DRAM_WRITE_ALIGNMENT_BYTES) - 1;
        debug_sanitize_post_noc_addr_and_hang(
            noc_id, noc_addr, l1_addr, noc_len, multicast, dir, DEBUG_SANITIZE_NOC_TARGET, debug_valid_dram_addr(noc_local_addr, noc_len));
#ifndef ARCH_GRAYSKULL
    } else if (core_type == AddressableCoreType::ETH) {
        if (!debug_valid_reg_addr(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(
                noc_id, noc_addr, l1_addr, noc_len, multicast, dir, DEBUG_SANITIZE_NOC_TARGET, debug_valid_eth_addr(noc_local_addr, noc_len));
        }
#endif
    } else if (core_type == AddressableCoreType::TENSIX) {
        if (!debug_valid_reg_addr(noc_local_addr, noc_len) && !debug_valid_worker_addr(noc_local_addr, noc_len)) {
            debug_sanitize_post_noc_addr_and_hang(
                noc_id, noc_addr, l1_addr, noc_len, multicast, dir, DEBUG_SANITIZE_NOC_TARGET, debug_valid_worker_addr(noc_local_addr, noc_len));
        }
    } else {
        // Bad XY
        debug_sanitize_post_noc_addr_and_hang(
            noc_id, noc_addr, l1_addr, noc_len, multicast, dir, DEBUG_SANITIZE_NOC_TARGET, DebugSanitizeNocTargetInvalidXY);
    }

    return alignment_mask;
}

void debug_sanitize_noc_and_worker_addr(
    uint8_t noc_id,
    uint64_t noc_addr,
    uint32_t worker_addr,
    uint32_t len,
    debug_sanitize_noc_cast_t multicast,
    debug_sanitize_noc_dir_t dir) {
    // Check noc addr, get any extra alignment req for worker.
    uint32_t alignment_mask = debug_sanitize_noc_addr(noc_id, noc_addr, worker_addr, len, multicast, dir);

    // Check worker addr and alignment, but these don't apply to regs.
    if (!debug_valid_reg_addr(worker_addr, len)) {
        debug_sanitize_post_noc_addr_and_hang(
            noc_id, noc_addr, worker_addr, len, multicast, dir, DEBUG_SANITIZE_NOC_LOCAL, debug_valid_worker_addr(worker_addr, len));

        if ((worker_addr & alignment_mask) != (noc_addr & alignment_mask)) {
            debug_sanitize_post_noc_addr_and_hang(
                noc_id, noc_addr, worker_addr, len, multicast, dir, DEBUG_SANITIZE_NOC_TARGET, DebugSanitizeNocAlignment);
        }
    }
    #error
}

// TODO: Clean these up with #7453
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_FROM_STATE(noc_id)                                   \
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(                                                         \
        noc_id,                                                                                  \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) |   \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID) << 32) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO)),       \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO),                        \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE));
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_FROM_STATE(noc_id)                               \
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(                                                     \
        noc_id,                                                                               \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_MID) << 32) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO)),     \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO),                    \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE));
#define DEBUG_SANITIZE_NOC_ADDR_FROM_STATE(noc_id, cmd_buf)                                   \
    DEBUG_SANITIZE_NOC_ADDR(                                                                  \
        noc_id,                                                                               \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, cmd_buf, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, cmd_buf, NOC_TARG_ADDR_MID) << 32) |      \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, cmd_buf, NOC_TARG_ADDR_LO)),              \
        4);
#define DEBUG_SANITIZE_NOC_ADDR(noc_id, a, l)                                                      \
    debug_sanitize_noc_addr(noc_id, a, 0, l, DEBUG_SANITIZE_NOC_UNICAST, DEBUG_SANITIZE_NOC_READ); \
    LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_TRANSACTION(noc_id, noc_a, worker_a, l, multicast, dir)  \
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, multicast, dir); \
    LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_id, noc_a, worker_a, l)                                                  \
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, DEBUG_SANITIZE_NOC_UNICAST, DEBUG_SANITIZE_NOC_READ); \
    LOG_LEN(l);                                                                                                  \
    debug_insert_delay((uint8_t)TransactionRead);
#define DEBUG_SANITIZE_NOC_MULTI_READ_TRANSACTION(noc_id, noc_a, worker_a, l)                                              \
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, DEBUG_SANITIZE_NOC_MULTICAST, DEBUG_SANITIZE_NOC_READ); \
    LOG_LEN(l);                                                                                                    \
    debug_insert_delay((uint8_t)TransactionRead);
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id, noc_a, worker_a, l)                                                  \
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, DEBUG_SANITIZE_NOC_UNICAST, DEBUG_SANITIZE_NOC_WRITE); \
    LOG_LEN(l);                                                                                                   \
    debug_insert_delay((uint8_t)TransactionWrite)
#define DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_id, noc_a, worker_a, l)                                              \
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, DEBUG_SANITIZE_NOC_MULTICAST, DEBUG_SANITIZE_NOC_WRITE); \
    LOG_LEN(l);                                                                                                     \
    debug_insert_delay((uint8_t)TransactionWrite);

#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a)         \
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(                                                                    \
        noc_id,                                                                                             \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID) << 32) | \
            noc_a_lower, \
        worker_a,                                                                                           \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE));
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_STATE(noc_id, noc_a_lower, worker_a, l)               \
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(                                                                    \
        noc_id,                                                                                             \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID) << 32) | \
            noc_a_lower, \
        worker_a,                                                                                           \
        l);
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a)           \
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(                                                                      \
        noc_id,                                                                                                \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_MID) << 32) | \
            noc_a_lower, \
        worker_a,                                                                                              \
        NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE));
#define DEBUG_INSERT_DELAY(transaction_type) debug_insert_delay(transaction_type)

// Delay for debugging purposes
inline void debug_insert_delay(uint8_t transaction_type) {
#if defined(WATCHER_DEBUG_DELAY)
    debug_insert_delays_msg_t tt_l1_ptr *v = GET_MAILBOX_ADDRESS_DEV(watcher.debug_insert_delays);

    bool delay = false;
    switch (transaction_type) {
        case TransactionRead: delay = (v[0].read_delay_riscv_mask & (1 << debug_get_which_riscv())) != 0; break;
        case TransactionWrite: delay = (v[0].write_delay_riscv_mask & (1 << debug_get_which_riscv())) != 0; break;
        case TransactionAtomic: delay = (v[0].atomic_delay_riscv_mask & (1 << debug_get_which_riscv())) != 0; break;
        default: break;
    }
    if (delay) {
        // WATCHER_DEBUG_DELAY is a compile time constant passed with -D
        riscv_wait (WATCHER_DEBUG_DELAY);
        v[0].feedback |= (1 << transaction_type); // Mark that we have delayed on this transaction type
    }
#endif  // WATCHER_DEBUG_DELAY
}

#else  // !WATCHER_ENABLED

#define DEBUG_SANITIZE_NOC_ADDR(noc_id, a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_TRANSACTION(noc_id, noc_a, worker_a, l, multicast, dir) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_id, noc_a, worker_a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_MULTI_READ_TRANSACTION(noc_id, noc_a, worker_a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id, noc_a, worker_a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_id, noc_a, worker_a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a) \
    LOG_READ_LEN_FROM_STATE(noc_id)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_STATE(noc_id, noc_a_lower, worker_a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a) \
    LOG_WRITE_LEN_FROM_STATE(noc_id)
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_FROM_STATE(noc_id)
#define DEBUG_SANITIZE_NOC_ADDR_FROM_STATE(noc_id, cmd_buf)
#define DEBUG_INSERT_DELAY(transaction_type)

#endif  // WATCHER_ENABLED
