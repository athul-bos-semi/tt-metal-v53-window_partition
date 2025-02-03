// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TODO: WH only, need to move this to generated code path for BH support
uint16_t eth_chan_to_noc_xy[2][14] __attribute__((used)) = {
    {
        // noc=0
        (((1 << NOC_ADDR_NODE_ID_BITS) | 1) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 2) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 3) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 4) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 5) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 6) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 7) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 10) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 11) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 12) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 13) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 14) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 15) << NOC_COORD_REG_OFFSET),
        (((1 << NOC_ADDR_NODE_ID_BITS) | 16) << NOC_COORD_REG_OFFSET),
    },
    {
        // noc=1
        (((10 << NOC_ADDR_NODE_ID_BITS) | 15) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 14) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 13) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 12) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 11) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 10) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 9) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 6) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 5) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 4) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 3) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 2) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 1) << NOC_COORD_REG_OFFSET),
        (((10 << NOC_ADDR_NODE_ID_BITS) | 0) << NOC_COORD_REG_OFFSET),
    },
};
