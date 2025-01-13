// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::blackhole {

// If you are trying to include this file and you aren't hal...you are doing something wrong

HalCoreInfoType create_tensix_mem_map();
HalCoreInfoType create_active_eth_mem_map();
HalCoreInfoType create_idle_eth_mem_map();

}  // namespace tt::tt_metal::blackhole
