// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dev_msgs.h
//
// Contains the structures/values uses in mailboxes to send messages to/from
// host and device and across brisc/ncrisc/trisc
//

#pragma once

#include "noc/noc_parameters.h"

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t)&(((mailboxes_t *)MEM_MAILBOX_BASE)->x))
#define GET_MAILBOX_ADDRESS_DEV(x) (&(((mailboxes_t tt_l1_ptr *)MEM_MAILBOX_BASE)->x))

// Messages for host to tell brisc to go
constexpr uint32_t RUN_MSG_INIT = 0x40;
constexpr uint32_t RUN_MSG_GO   = 0x80;
constexpr uint32_t RUN_MSG_DONE = 0;

// 0x80808000 is a micro-optimization, calculated with 1 riscv insn
constexpr uint32_t RUN_SYNC_MSG_INIT = 0x40;
constexpr uint32_t RUN_SYNC_MSG_GO   = 0x80;
constexpr uint32_t RUN_SYNC_MSG_DONE = 0;
constexpr uint32_t RUN_SYNC_MSG_ALL_TRISCS_GO = 0x80808000;
constexpr uint32_t RUN_SYNC_MSG_ALL_GO = 0x80808080;
constexpr uint32_t RUN_SYNC_MSG_ALL_SLAVES_DONE = 0;

struct ncrisc_halt_msg_t {
    volatile uint32_t resume_addr;
    volatile uint32_t stack_save;
};

enum dispatch_mode {
    DISPATCH_MODE_DEV,
    DISPATCH_MODE_HOST,
};

struct launch_msg_t {  // must be cacheline aligned
    volatile uint16_t brisc_watcher_kernel_id;
    volatile uint16_t ncrisc_watcher_kernel_id;
    volatile uint16_t triscs_watcher_kernel_id;
    volatile uint16_t ncrisc_kernel_size16; // size in 16 byte units

    // TODO(agrebenisan): This must be added in to launch_msg_t
    // volatile uint16_t dispatch_core_x;
    // volatile uint16_t dispatch_core_y;
    volatile uint8_t  mode;
    volatile uint8_t  brisc_noc_id;
    volatile uint8_t  enable_brisc;
    volatile uint8_t  enable_ncrisc;
    volatile uint8_t  enable_triscs;
    volatile uint8_t  max_cb_index;
    volatile uint8_t  enable_erisc;
    volatile uint8_t  run;  // must be in last cacheline of this msg
};

struct slave_sync_msg_t {
    union {
        volatile uint32_t all;
        struct {
            volatile uint8_t ncrisc; // ncrisc must come first, see ncrisc-halt.S
            volatile uint8_t trisc0;
            volatile uint8_t trisc1;
            volatile uint8_t trisc2;
        };
    };
};

constexpr int num_status_bytes_per_riscv = 4;
struct debug_status_msg_t {
    volatile uint8_t status[num_status_bytes_per_riscv];
};

struct debug_sanitize_noc_addr_msg_t {
    volatile uint64_t addr;
    volatile uint32_t len;
    volatile uint16_t which;
    volatile uint16_t invalid;
};

enum debug_sanitize_noc_invalid_enum {
    // 0 and 1 are a common stray values to write, so don't use those
    DebugSanitizeNocInvalidOK         = 2,
    DebugSanitizeNocInvalidL1         = 3,
    DebugSanitizeNocInvalidUnicast    = 4,
    DebugSanitizeNocInvalidMulticast  = 5,
};

constexpr int num_riscv_per_core = 5;
struct mailboxes_t {
    struct ncrisc_halt_msg_t ncrisc_halt;
    volatile uint32_t l1_barrier;
    struct launch_msg_t launch;
    struct slave_sync_msg_t slave_sync;
    struct debug_status_msg_t debug_status[num_riscv_per_core];
    struct debug_sanitize_noc_addr_msg_t sanitize_noc[NUM_NOCS];
};

#ifndef TENSIX_FIRMWARE
// Validate assumptions on mailbox layout on host compile
static_assert((MEM_MAILBOX_BASE + offsetof(mailboxes_t, launch)) % 16 == 0);
static_assert(MEM_MAILBOX_BASE + offsetof(mailboxes_t, slave_sync.ncrisc) == MEM_SLAVE_RUN_MAILBOX_ADDRESS);
static_assert(MEM_MAILBOX_BASE + offsetof(mailboxes_t, ncrisc_halt.stack_save) == MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS);
static_assert(MEM_MAILBOX_BASE + sizeof(mailboxes_t) < MEM_MAILBOX_END);
#endif
