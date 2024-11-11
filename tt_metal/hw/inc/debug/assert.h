// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "watcher_common.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT) && !defined(FORCE_WATCHER_OFF)

void assert_and_hang(uint32_t line_num) {
    // Write the line number into the memory mailbox for host to read.
    debug_assert_msg_t tt_l1_ptr *v = GET_MAILBOX_ADDRESS_DEV(get_mailbox_base(), watcher.assert_status);
    if (v->tripped == DebugAssertOK) {
        v->line_num = line_num;
        v->tripped = DebugAssertTripped;
        v->which = debug_get_which_riscv();
    }

    // Hang, or in the case of erisc, early exit.
#if defined(COMPILE_FOR_ERISC)
    // Update launch msg to show that we've exited. This is required so that the next run doesn't think there's a kernel
    // still running and try to make it exit.
    tt_l1_ptr go_msg_t *go_message_ptr = GET_MAILBOX_ADDRESS_DEV(get_mailbox_base(), go_message);
    go_message_ptr->signal = RUN_MSG_DONE;

    // This exits to base FW
    internal_::disable_erisc_app();
    erisc_exit();
#endif

    while(1) { ; }
}

// The do... while(0) in this macro allows for it to be called more flexibly, e.g. in an if-else
// without {}s.
#define ASSERT(condition) do{ if (not (condition)) assert_and_hang(__LINE__); } while(0)

#else // !WATCHER_ENABLED

#define ASSERT(condition)

#endif // WATCHER_ENABLED
