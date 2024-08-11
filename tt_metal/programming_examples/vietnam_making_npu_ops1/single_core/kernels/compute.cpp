#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    // TODO. get runtime arguments.
    uint32_t arg = 0;
    const auto cb0_id = get_arg_val<uint32_t>(arg++);
    const auto cb1_id = get_arg_val<uint32_t>(arg++);
    const auto num_tiles = get_arg_val<uint32_t>(arg++);

    constexpr auto dst0 = 0;
    constexpr auto first = 0;
    constexpr auto onetile = 1;

    unary_op_init_common(cb0_id, cb1_id);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        // TODO 1. Copy one tile in cb0 to dst0 register with cb_wait_front and cb_pop_front.

        // TODO 2. Do relu operation on dst0.

        tile_regs_commit();

        tile_regs_wait();
        // TODO 3. Copy dst0 register to cb1 with cb_reserve_back and cb_push_back.

        cb_push_back(cb1_id, onetile);

        tile_regs_release();
    }

    UNPACK(DPRINT << "UNPACK END" << ENDL());
    MATH(DPRINT << "MATH END" << ENDL());
    PACK(DPRINT << "PACK END" << ENDL());
}
}  // namespace NAMESPACE
