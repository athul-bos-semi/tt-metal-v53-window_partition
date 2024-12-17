// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0
///

#include <cstdint>
#include <functional>
#include <limits>
#include <vector>
#include <array>
#include <ranges>
#include "common/core_coord.hpp"
#include "common/logger.hpp"
#include "device/device.hpp"
#include "kernels/kernel_types.hpp"
#include "span.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/erisc_datamover_builder.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operation.hpp"

// For reduction op
#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "tt_metal/impl/buffers/global_semaphore.hpp"

/*
 * This file contains the program factory for reduce scatter operation implemented on line (and soon, ring) topologies.
 * The current implementation is fairly memory inefficient, however, even when optimized the general approach is as
 follows:
 *
 * Lo
 *
 *   IN 0     IN 1     IN 2     IN 3            OUT 0    OUT 1    OUT 2    OUT 3
 *   C0       C1       C2       C3              C0       C1       C2       C3
 *  ┌────┐   ┌────┐   ┌────┐   ┌────┐          ┌────┐   ......   ......   ......
 *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
 *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
 *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
 *  ├────┤   ├────┤   ├────┤   ├────┤          └────┘   ┌────┐   ......   ......
 *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
 *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
 *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
 *  ├────┤   ├────┤   ├────┤   ├────┤  ────►   ......   └────┘   ┌────┐   ......
 *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
 *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
 *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
 *  ├────┤   ├────┤   ├────┤   ├────┤          ......   ......   └────┘   ┌────┐
 *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
 *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
 *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
 *  └────┘   └────┘   └────┘   └────┘          ......   ......   ......   └────┘
 *
 *
 *
 *



 *
 *      ┌────┐      ┌────┐     ┌────┐     ┌────┐
 *      ├─►+◄┼──────┼─ +◄┼─────┼──+◄┼─────┼──  │
 *      │    │      │  ▲ │     │  ▲ │     │    │
 *      │    │      │  └ │     │  └ │     │    │
 *      ┼────┼      ┼────┼     ┼────┼     ┼────┼
 *      │    │      │ ┌  │     │  ┌ │     │    │
 *      │    │      │ ▼  │     │  ▼ │     │    │
 *      │ ───┼──────┼►+◄─┼─────┼──+◄┼─────┼──  │
 *      ┼────┼      ┼────┼     ┼────┼     ┼────┼
 *      │    │      │ ┌  │     │  ┌ │     │    │
 *      │    │      │ ▼  │     │  ▼ │     │    │
 *      │  ──┼──────┼►+──┼─────┼─►+◄┼─────┼──  │
 *      ┼────┼      ┼────┼     ┼────┼     ┼────┼
 *      │    │      │ ┌  │     │ ┌  │     │ ┌  │
 *      │    │      │ ▼  │     │ ▼  │     │ ▼  │
 *      │  ──┼──────┼►+──┼─────┼►+ ─┼─────┼►+  │
 *      └────┘      └────┘     └────┘     └────┘
 *
 *
 */

namespace ttnn::ccl::reduce_scatter_detail {

using ttnn::ccl::Shape4D;
using ttnn::ccl::cmd::CclCommandTensor;

enum fabric_lifetime_mode {
    // The fabric's lifetime exceed (before and after) the lifetime of the op
    // so the op should not in any way manage fabric lifetime
    PERSISTENT,

    // The fabric is brought up and torn down for each op invocation
    TRANSIENT
};

enum LineDirection { FORWARD, BACKWARD };
static_assert(
    static_cast<size_t>(LineDirection::FORWARD) ==
    static_cast<size_t>(ttnn::ccl::EdmLineFabricOpInterface::Direction::FORWARD));
static_assert(
    static_cast<size_t>(LineDirection::BACKWARD) ==
    static_cast<size_t>(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD));

// TODO: promote to header

struct ReduceScatterCircularBuffers {
    uint32_t reader_to_writer_shortcut_cb = -1;
    uint32_t reader_to_math_operand0_cb = -1;
    uint32_t reader_to_math_operand1_cb = -1;
    uint32_t math_to_writer_cb = -1;
    CBHandle reader_to_writer_shortcut_cb_handle = -1;
    CBHandle reader_to_math_operand0_cb_handle = -1;
    CBHandle reader_to_math_operand1_cb_handle = -1;
    CBHandle math_to_writer_cb_handle = -1;
};

struct CircularBufferSpec {
    size_t cb_size = 0;
    size_t page_size = 0;
    uint32_t cb_index = 0;
    tt::DataFormat df = tt::DataFormat::Invalid;
};

struct ReduceScatterKernelHandles {
    KernelHandle reader = -1;
    KernelHandle math = -1;
    KernelHandle writer = -1;
};

// We really need something like a graph here to describe the dependencies generically but for
// now we keep it very simple and constrained
struct TensorSyncSpec {
    static constexpr int UNINITIALIZED_DEST_NOC = -1;
    struct target_rect {
        int dest_noc0_x_start = UNINITIALIZED_DEST_NOC;
        int dest_noc0_y_start = UNINITIALIZED_DEST_NOC;
        int dest_noc0_x_end = UNINITIALIZED_DEST_NOC;
        int dest_noc0_y_end = UNINITIALIZED_DEST_NOC;
    };
    // always equal to number of slices for now
    std::vector<std::variant<uint32_t, const GlobalSemaphore*>> semaphore_ids;
    std::vector<size_t> completion_target_value_per_semaphore;
    std::vector<target_rect> targets;

    ttnn::ccl::cmd::CclCommandCoreDescriptorTypeMcast get_target(size_t i) const {
        return ttnn::ccl::cmd::CclCommandCoreDescriptorTypeMcast{
            static_cast<uint32_t>(targets.at(i).dest_noc0_x_start),
            static_cast<uint32_t>(targets.at(i).dest_noc0_y_start),
            static_cast<uint32_t>(targets.at(i).dest_noc0_x_end),
            static_cast<uint32_t>(targets.at(i).dest_noc0_y_end)};
    }

    size_t num_semaphores() const { return semaphore_ids.size(); }

    std::variant<uint32_t, const GlobalSemaphore*> const& get_tensor_sync_semaphore(size_t slice_index) const {
        TT_FATAL(
            slice_index < semaphore_ids.size(),
            "Internal error. Requested semaphore id for slice index that does not exist");
        return semaphore_ids.at(slice_index);
    }
};

struct WorkerCoreBundle {
    CoreRangeSet all_worker_cores;
    CoreRangeSet final_reducers;
    std::array<CoreRangeSet, 2> partial_reducers;

    std::vector<CoreCoord> all_worker_cores_vec;
    std::vector<CoreCoord> final_reducers_vec;
    std::array<std::vector<CoreCoord>, 2> partial_reducers_vec;
};

struct ProgramTensorsBundle {
    Tensor const* input_tensor = nullptr;
    std::optional<TensorSyncSpec> input_tensor_sync;
    Tensor* local_output_tensor = nullptr;
    std::optional<TensorSyncSpec> local_output_sync;
    std::array<Tensor*, 2> input_tensor_from_remote = {nullptr, nullptr};
    std::array<TensorSyncSpec, 2> input_tensor_from_remote_sync;
    std::array<Tensor*, 2> remote_output = {nullptr, nullptr};
    std::array<TensorSyncSpec, 2> remote_output_sync;
    std::array<Tensor*, 2> local_output_partial = {nullptr, nullptr};
    std::array<TensorSyncSpec, 2> local_output_partial_sync;

    static Tensor* build_handle(Tensor& tensor) { return &tensor; }
    static Tensor const* build_handle(Tensor const& tensor) { return &tensor; }
    static Tensor* build_handle(std::optional<Tensor>& tensor) {
        return tensor.has_value() ? &tensor.value() : nullptr;
    }
};

static ReduceScatterCircularBuffers create_worker_circular_buffers(
    tt::tt_metal::Program& program,
    CoreRangeSet const& worker_core_range,
    CircularBufferSpec const& shortcut_cb_spec,
    CircularBufferSpec const& reader_to_math0_cb_spec,
    CircularBufferSpec const& reader_to_math1_cb_spec,
    CircularBufferSpec const& math_to_writer_cb_spec) {
    TT_FATAL(
        shortcut_cb_spec.cb_size % shortcut_cb_spec.page_size == 0,
        "Shortcut circular buffer size must be a multiple of the page size");
    TT_FATAL(
        reader_to_math0_cb_spec.cb_size % reader_to_math0_cb_spec.page_size == 0,
        "Reader to math circular buffer size must be a multiple of the page size");
    TT_FATAL(
        reader_to_math1_cb_spec.cb_size % reader_to_math1_cb_spec.page_size == 0,
        "Reader to math circular buffer size must be a multiple of the page size");
    TT_FATAL(
        math_to_writer_cb_spec.cb_size % math_to_writer_cb_spec.page_size == 0,
        "Math to writer circular buffer size must be a multiple of the page size");

    auto generate_circular_buffer = [&program, &worker_core_range](CircularBufferSpec const& cb_spec) -> CBHandle {
        tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(cb_spec.cb_size, {{cb_spec.cb_index, cb_spec.df}})
                .set_page_size(cb_spec.cb_index, cb_spec.page_size);
        CBHandle cb_handle = CreateCircularBuffer(program, worker_core_range, cb_config);
        return cb_handle;
    };

    return ReduceScatterCircularBuffers{
        shortcut_cb_spec.cb_index,
        reader_to_math0_cb_spec.cb_index,
        reader_to_math1_cb_spec.cb_index,
        math_to_writer_cb_spec.cb_index,
        generate_circular_buffer(shortcut_cb_spec),
        generate_circular_buffer(reader_to_math0_cb_spec),
        generate_circular_buffer(reader_to_math1_cb_spec),
        generate_circular_buffer(math_to_writer_cb_spec)};
}

static ReduceScatterCircularBuffers create_worker_circular_buffers(
    tt::tt_metal::Program& program,
    CoreRangeSet const& worker_core_range,

    const tt::CBIndex math_in0_cb,
    const tt::CBIndex math_in1_cb,
    const tt::CBIndex math_out_cb,
    const tt::CBIndex pass_through_cb,
    size_t fabric_buffer_size_pages,
    size_t page_size) {
    size_t buffer_depth_multiplier = 3;
    auto cb_handles = create_worker_circular_buffers(
        program,
        worker_core_range,
        CircularBufferSpec{
            buffer_depth_multiplier * fabric_buffer_size_pages * page_size,
            page_size,
            pass_through_cb,
            tt::DataFormat::Float32},
        CircularBufferSpec{
            buffer_depth_multiplier * fabric_buffer_size_pages * page_size,
            page_size,
            math_in0_cb,
            tt::DataFormat::Float32},
        CircularBufferSpec{
            buffer_depth_multiplier * fabric_buffer_size_pages * page_size,
            page_size,
            math_in1_cb,
            tt::DataFormat::Float32},
        CircularBufferSpec{
            buffer_depth_multiplier * fabric_buffer_size_pages * page_size,
            page_size,
            math_out_cb,
            tt::DataFormat::Float32});

    TT_FATAL(cb_handles.math_to_writer_cb != -1, "Math to writer circular buffer handle is invalid");
    TT_FATAL(cb_handles.reader_to_math_operand0_cb != -1, "Reader to math0 circular buffer handle is invalid");
    TT_FATAL(cb_handles.reader_to_math_operand1_cb != -1, "Reader to math1 circular buffer handle is invalid");
    TT_FATAL(
        cb_handles.reader_to_writer_shortcut_cb != -1, "Reader to writer shortcut circular buffer handle is invalid");
    return cb_handles;
}

template <typename T>
static std::vector<T> vslice(std::vector<T> const& vec, std::size_t start, std::size_t end_inclusive) {
    assert(end_inclusive < vec.size());
    assert(start < vec.size());
    std::vector<T> output;
    if (start > end_inclusive) {
        size_t n_elem = start - end_inclusive + 1;
        output.reserve(n_elem);
        std::copy(
            vec.rbegin() + (vec.size() - 1 - start),
            vec.rbegin() + (vec.size() - 1 - start + n_elem),
            std::back_inserter(output));

    } else {
        output.reserve(end_inclusive - start + 1);
        std::copy(vec.begin() + start, vec.begin() + end_inclusive + 1, std::back_inserter(output));
    }
    return output;
}

class LineTopology {
public:
    LineTopology(size_t line_size, size_t line_index) : _line_size(line_size), _line_index(line_index) {}

    bool is_first_device_in_line(LineDirection direction) const {
        if (direction == LineDirection::FORWARD) {
            return _line_index == 0;
        } else {
            TT_ASSERT(direction == LineDirection::BACKWARD);
            return _line_index == _line_size - 1;
        }
    }
    bool is_last_device_in_line(LineDirection direction) const {
        if (direction == LineDirection::BACKWARD) {
            return _line_index == 0;
        } else {
            TT_ASSERT(direction == LineDirection::FORWARD);
            return _line_index == _line_size - 1;
        }
    }

    bool is_at_end_of_line() const { return _line_index == 0 || _line_index == _line_size - 1; }

    size_t line_size() const { return _line_size; }

    size_t line_index() const { return _line_index; }

    ttnn::ccl::Topology topology() const { return ttnn::ccl::Topology::Linear; }

private:
    size_t _line_size;
    size_t _line_index;
};

struct TensorSyncBundle {
    const Tensor* tensor;
    std::optional<TensorSyncSpec> sync_spec;
};

struct ReaderCircularBufferIds {
    uint32_t pass_through;
    uint32_t math_in0;
    uint32_t math_in1;
};

struct WriterCircularBufferIds {
    uint32_t pass_through;
    uint32_t math_out;
};
struct FinalReducerReaderCircularBufferIds {
    uint32_t math_in0;
    uint32_t math_in1;
};
struct FinalReducerWriterCircularBufferIds {
    uint32_t math_out;
};
struct LineStartReaderCircularBufferIds {
    uint32_t pass_through;
};
struct LineStartWriterCircularBufferIds {
    uint32_t pass_through;
};
struct LineEndReaderCircularBufferIds {
    uint32_t math_in0;
    uint32_t math_in1;
};
struct LineEndWriterCircularBufferIds {
    uint32_t math_out;
};

struct AllReduceScatterCircularBufferIds {
    ReaderCircularBufferIds partial_reducer_reader;
    WriterCircularBufferIds partial_reducer_writer;
    FinalReducerReaderCircularBufferIds final_reducer_reader;
    FinalReducerWriterCircularBufferIds final_reducer_writer;
    LineStartReaderCircularBufferIds line_start_reader;
    LineStartWriterCircularBufferIds line_start_writer;
    LineEndReaderCircularBufferIds line_end_reader;
    LineEndWriterCircularBufferIds line_end_writer;
};

struct WorkerCommandStreams {
    std::unordered_map<CoreCoord, ttnn::ccl::cmd::CclHostLowLevelCommandSequence> reader_cmds0;
    std::unordered_map<CoreCoord, ttnn::ccl::cmd::CclHostLowLevelCommandSequence> reader_cmds1;
    std::unordered_map<CoreCoord, ttnn::ccl::cmd::CclHostLowLevelCommandSequence> writer_cmds0;
    std::unordered_map<CoreCoord, ttnn::ccl::cmd::CclHostLowLevelCommandSequence> writer_cmds1;
};

struct ReduceScatterBuilderConfig {
    std::reference_wrapper<Program> program;
    Device* device;
    Device* forward_device;
    Device* backward_device;
    std::reference_wrapper<ttnn::ccl::EdmLineFabricOpInterface> fabric;
    std::reference_wrapper<ProgramTensorsBundle> all_tensors;
    std::reference_wrapper<ReduceScatterKernelHandles> kernel_ids;
    std::reference_wrapper<const AllReduceScatterCircularBufferIds> all_cbs;
    std::reference_wrapper<const LineTopology> topology_config;
    std::reference_wrapper<const WorkerCoreBundle> worker_cores;
    size_t page_size = std::numeric_limits<size_t>::max();
    size_t pages_per_cb_packet = std::numeric_limits<size_t>::max();
    size_t dim = std::numeric_limits<size_t>::max();
};

/*
static WorkerCoreBundle select_worker_cores_for_line_topology(size_t num_links, Device *device) {

    auto build_all_workers_list = [](CoreRangeSet const& available_cores, size_t total_cores_needed, std::vector<CoreCoord> &all_cores_out) {

        for (const auto& cr : available_cores.ranges()) {
            auto start = cr.start_coord;
            auto end = cr.end_coord;
            for (size_t y = start.y; y <= end.y; y++) {
                for (size_t x = start.x; x <= end.x; x++) {
                    all_cores_out.push_back(CoreCoord(x, y));
                    if (all_cores_out.size() == total_cores_needed) {
                        return;
                    }
                }
            }
        }
    };

    static constexpr std::size_t num_directions_per_line = 2;
    WorkerCoreBundle worker_cores;
    size_t current_chunk = 0;

    constexpr size_t num_final_reducers_per_link = 1;
    constexpr size_t per_link_num_workers_needed = num_directions_per_line + num_final_reducers_per_link;
    const size_t total_cores_needed = per_link_num_workers_needed * num_links;
    const auto available_cores =
        device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().at(0));
    if (available_cores.num_cores() < total_cores_needed) {
        log_warning(
            tt::LogOp,
            "AllGather is being launched on a subdevice with fewer worker cores available than ideal. Ideally {} "
            "cores are available ({} per link and {} links) are made available but only {} are available. This may "
            "lead to performance loss.",
            total_cores_needed,
            per_link_num_workers_needed,
            num_links,
            available_cores.num_cores());
        TT_THROW("Reduce scatter async currently doesn't support running with fewer than preferred number of workers");
    }
    std::vector<CoreCoord> all_cores;
    all_cores.reserve(total_cores_needed);
    build_all_workers_list(available_cores, total_cores_needed, all_cores);

    auto add_workers = [&num_links](std::vector<CoreCoord>::iterator &worker_iter, CoreRangeSet& cores) {
        for (size_t l = 0; l < num_links; l++) {
            cores = cores.merge(CoreRangeSet(CoreRange(*worker_iter)));
            worker_iter++;
        }
    };

    auto worker_coord_iter = all_cores.begin();
    for (size_t d = 0; d < num_directions_per_line; d++) {
        add_workers(worker_coord_iter, worker_cores.partial_reducers[d]);
    }
    add_workers(worker_coord_iter, worker_cores.final_reducers);

    // Merge them all into the global set for convenience anywhere we want to access all worker cores easily
    for (size_t d = 0; d < num_directions_per_line; d++) {
        worker_cores.all_worker_cores = worker_cores.all_worker_cores.merge(worker_cores.partial_reducers[d]);
    }
    worker_cores.all_worker_cores = worker_cores.all_worker_cores.merge(worker_cores.final_reducers);
    log_trace(tt::LogOp, "Worker cores: ", worker_cores.all_worker_cores);

    worker_cores.all_worker_cores_vec = corerange_to_cores(worker_cores.all_worker_cores, std::nullopt, true);
    worker_cores.final_reducers_vec = corerange_to_cores(worker_cores.final_reducers, std::nullopt, true);
    for (size_t d = 0; d < num_directions_per_line; d++) {
        worker_cores.partial_reducers_vec[d] = corerange_to_cores(worker_cores.partial_reducers[d], std::nullopt, true);
    }

    return worker_cores;
}
*/


/*
 * Core range sets for line topology
 * BORROWED FROM REDUCE SCATTER but modified a fair bit
 * TODO: COMMONIZE
 */
static WorkerCoreBundle select_worker_cores_for_line_topology(size_t num_links) {
    static constexpr std::size_t num_directions_per_line = 2;
    WorkerCoreBundle worker_cores;
    size_t current_chunk = 0;
    for (size_t d = 0; d < num_directions_per_line; d++) {
        worker_cores.partial_reducers[d] =
            CoreRangeSet(CoreRange(CoreCoord(0, current_chunk), CoreCoord(num_links - 1, current_chunk)));
        current_chunk++;
    }
    worker_cores.final_reducers =
        CoreRangeSet(CoreRange(CoreCoord(0, current_chunk), CoreCoord(num_links - 1, current_chunk)));
    current_chunk++;

    // Merge them all into the global set for convenience anywhere we want to access all worker cores easily
    for (size_t d = 0; d < num_directions_per_line; d++) {
        worker_cores.all_worker_cores = worker_cores.all_worker_cores.merge(worker_cores.partial_reducers[d]);
    }
    worker_cores.all_worker_cores = worker_cores.all_worker_cores.merge(worker_cores.final_reducers);
    log_trace(tt::LogOp, "Worker cores: ", worker_cores.all_worker_cores);

    worker_cores.all_worker_cores_vec = corerange_to_cores(worker_cores.all_worker_cores, std::nullopt, true);
    worker_cores.final_reducers_vec = corerange_to_cores(worker_cores.final_reducers, std::nullopt, true);
    for (size_t d = 0; d < num_directions_per_line; d++) {
        worker_cores.partial_reducers_vec[d] = corerange_to_cores(worker_cores.partial_reducers[d], std::nullopt, true);
    }

    return worker_cores;
}

/*
 * Returns 1 or 2 core range sets. Typically returns only one but in the case of a line reduce scatter where we are at
 * the end of the line, then we must split the core range in half (and return 2), one for each direction where half the
 * cores will invoke the ccl::send kernel to implement the start of the line and the others will invoke the typical
 * reduce scatter worker kernels. BORROWED FROM REDUCE SCATTER
 * TODO: COMMONIZE
 */
static WorkerCoreBundle select_worker_cores(ttnn::ccl::Topology const topology, size_t num_links) {
    switch (topology) {
        case ttnn::ccl::Topology::Linear: return select_worker_cores_for_line_topology(num_links);

        case ttnn::ccl::Topology::Ring:
            TT_THROW("Ring topology support not yet added to async reduce scatter");
            return WorkerCoreBundle{};

        default: TT_ASSERT(false, "Unsupported topology"); return WorkerCoreBundle{};
    };
}

static size_t compute_math_pages_from_tensor_slices(
    std::vector<ttnn::ccl::v2::TensorSlice> const& tensor_slices, size_t pages_per_cb_packet) {
    using namespace ttnn::ccl::cmd;

    auto get_slice_vol = [pages_per_cb_packet](ttnn::ccl::v2::TensorSlice const& slice) {
        return round_up(slice.worker_slice_shape.volume(), pages_per_cb_packet);
    };

    size_t total_num_pages = 0;
    for (auto const& s : tensor_slices) {
        total_num_pages += get_slice_vol(s);
    }

    return total_num_pages;
}

/*
 * Returns the reader, math, and writer kernels, respectively
 */
static ReduceScatterKernelHandles build_line_reduce_scatter_worker_ct(
    Program& program,
    ProgramTensorsBundle const& all_tensors,
    ReduceScatterCircularBuffers const& cb_handles,
    CoreRangeSet const& worker_core_range,
    ttnn::operations::binary::BinaryOpType reduce_op) {
    using namespace ttnn::ccl::worker_detail;

    // Summary:
    // == READER ==
    //  - First CB: shortcut to writer
    //  - Second CB: to math (local input)
    //  - Third CB: to math (remote input)

    static std::string const& receiver_kernel_path =
        "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader_two_input.cpp";
    static std::string const& forward_sender_kernel_path = receiver_kernel_path;
    static std::string const& reduce_kernel_path =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp";

    // Generate the reader kernel
    auto input_tensor_ptrs = std::vector<Tensor const*>{
        all_tensors.input_tensor,
        all_tensors.input_tensor_from_remote[0] != nullptr ? all_tensors.input_tensor_from_remote[0]
                                                           : all_tensors.input_tensor_from_remote[1]};
    TT_FATAL(
        input_tensor_ptrs[0] != nullptr && input_tensor_ptrs[1] != nullptr,
        "Internal error. Input tensor pointers are null");
    auto reader_kernel_id = generate_multi_command_stream_kernel_ct_args(
        program,
        // the CBs don't actuall matter for CT args - they will be removed as CT args in the near future
        {cb_handles.reader_to_math_operand1_cb /*cb_handles.reader_to_writer_shortcut_cb*/,
         cb_handles.reader_to_math_operand0_cb},
        input_tensor_ptrs,
        worker_core_range,
        ReaderDataMovementConfig{});

    // Generate the math/reducer kernel
    std::vector<uint32_t> compute_kernel_args = {};
    constexpr bool fp32_dest_acc_en = false;
    constexpr bool math_approx_mode = false;
    std::map<string, string> eltwise_defines = ttnn::operations::binary::utils::get_defines(reduce_op);
    auto math_kernel_id = tt::tt_metal::CreateKernel(
        program,
        reduce_kernel_path,
        worker_core_range,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = eltwise_defines});

    // Generate the sender kernel
    auto const output_tensor_ptrs = std::vector<Tensor const*>{
        all_tensors.remote_output[0] != nullptr ? all_tensors.remote_output[0] : all_tensors.remote_output[1],
        all_tensors.local_output_tensor};
    auto sender_kernel_id = generate_multi_command_stream_kernel_ct_args(
        program,
        {cb_handles.reader_to_writer_shortcut_cb, cb_handles.math_to_writer_cb},
        output_tensor_ptrs,
        worker_core_range,
        WriterDataMovementConfig{});

    return ReduceScatterKernelHandles{reader_kernel_id, math_kernel_id, sender_kernel_id};
}

static size_t get_page_size(const Tensor& tensor) {
    if (tensor.get_layout() == Layout::TILE) {
        auto dtype = tt::tt_metal::datatype_to_dataformat_converter(tensor.get_dtype());
        return tensor.get_tensor_spec().tile().get_tile_size(dtype);
    } else {
        return tensor.buffer()->page_size();
    }
}

static void validate_final_reducer_reader_worker_slices(
    std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> const& in0_worker_slices,
    std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> const& in1_worker_slices,
    std::optional<TensorSyncSpec> const& in0_sync,
    std::optional<TensorSyncSpec> const& in1_sync,
    size_t num_workers) {
    TT_FATAL(in0_sync.has_value(), "Internal error. Final reducer saw that in0 had not tensor synchronization info");
    TT_FATAL(in1_sync.has_value(), "Internal error. Final reducer saw that in1 had not tensor synchronization info");
    TT_FATAL(
        in0_worker_slices.size() == num_workers,
        "Internal error. Expected number of worker slices to match number of workers");
    TT_FATAL(
        in1_worker_slices.size() == num_workers,
        "Internal error. Expected number of worker slices to match number of workers");
    for (size_t w = 0; w < num_workers; w++) {
        TT_FATAL(in0_worker_slices[w].size() == 1, "Internal error. Expected only one slice per worker");
        TT_FATAL(in1_worker_slices[w].size() == 1, "Internal error. Expected only one slice per worker");
    }
}

static void generate_final_reducer_reader_worker_command_streams(
    ReduceScatterBuilderConfig& builder_config,
    TensorSyncBundle const& partial_output0_tensor_sync_bundle,
    TensorSyncBundle const& partial_output1_tensor_sync_bundle,
    WorkerCommandStreams& worker_command_streams_out,
    std::unordered_map<CoreCoord, size_t>& math_page_counts_out) {
    using namespace ttnn::ccl::cmd;
    using namespace ttnn::ccl::cmd::uops;
    using namespace ttnn::ccl::cmd::builder;

    auto const& all_tensors = builder_config.all_tensors.get();
    auto const& reader_cbs = builder_config.all_cbs.get().final_reducer_reader;
    size_t num_partial_reducer_workers =
        builder_config.worker_cores.get().partial_reducers[LineDirection::FORWARD].size();
    auto const& worker_cores = builder_config.worker_cores.get().final_reducers_vec;
    size_t num_workers = worker_cores.size();
    size_t pages_per_cb_packet = builder_config.pages_per_cb_packet;

    auto const in0_tensor_slice = generate_tensor_slices(1, *partial_output0_tensor_sync_bundle.tensor, 0).at(0);
    auto in0_worker_slices = split_tensor_slices_across_workers_page_aligned(num_workers, {in0_tensor_slice});
    auto const in1_tensor_slice = generate_tensor_slices(1, *partial_output1_tensor_sync_bundle.tensor, 0).at(0);
    auto in1_worker_slices = split_tensor_slices_across_workers_page_aligned(num_workers, {in1_tensor_slice});

    auto const& in0_sync = partial_output0_tensor_sync_bundle.sync_spec;
    auto const& in1_sync = partial_output1_tensor_sync_bundle.sync_spec;

    validate_final_reducer_reader_worker_slices(in0_worker_slices, in1_worker_slices, in0_sync, in1_sync, num_workers);
    for (size_t w = 0; w < num_workers; w++) {
        auto const& w_logical = worker_cores[w];
        auto& worker_command_stream0 = worker_command_streams_out.reader_cmds0[w_logical];
        // TODO: Semaphore inc/wait optimization
        worker_command_stream0 = {
            local_semaphore_wait(in0_sync.value().get_tensor_sync_semaphore(0), num_partial_reducer_workers),
            read_tensor_slice_to_cb(in0_worker_slices[w][0], reader_cbs.math_in0)};

        auto& worker_command_stream1 = worker_command_streams_out.reader_cmds1[w_logical];
        worker_command_stream1 = {
            local_semaphore_wait(in1_sync.value().get_tensor_sync_semaphore(0), num_partial_reducer_workers),
            read_tensor_slice_to_cb(in1_worker_slices[w][0], reader_cbs.math_in1)};

        math_page_counts_out[w_logical] =
            compute_math_pages_from_tensor_slices(in0_worker_slices[w], pages_per_cb_packet);
    }
}

static void generate_final_reducer_writer_worker_command_streams(
    ReduceScatterBuilderConfig& builder_config,
    // Should only have populated sync info if fused
    TensorSyncBundle const& output_tensor_sync_bundle,
    WorkerCommandStreams& worker_command_streams_out) {
    using namespace ttnn::ccl::cmd;
    using namespace ttnn::ccl::cmd::uops;
    using namespace ttnn::ccl::cmd::builder;

    auto from_math_cb = builder_config.all_cbs.get().final_reducer_writer.math_out;
    auto const& worker_cores = builder_config.worker_cores.get().final_reducers_vec;
    size_t num_workers = worker_cores.size();

    auto const tensor_slice = generate_tensor_slices(1, *output_tensor_sync_bundle.tensor, 0).at(0);
    auto worker_slices = split_tensor_slices_across_workers_page_aligned(num_workers, {tensor_slice});

    auto const& sync = output_tensor_sync_bundle.sync_spec;
    TT_FATAL(
        worker_slices.size() == num_workers,
        "Internal error. Expected number of worker slices to match number of workers");
    auto& writer_cmds = worker_command_streams_out.writer_cmds0;
    for (size_t w = 0; w < num_workers; w++) {
        auto const& w_logical = worker_cores[w];
        TT_FATAL(worker_slices[w].size() == 1, "Internal error. Expected only one slice per worker");
        writer_cmds[w_logical].push_back({local_write_cb_to_tensor_slice(worker_slices[w][0], from_math_cb)});
    }
}

static void compute_math_pages_from_per_worker_tensor_slices(
    std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> const& worker_slices,
    size_t pages_per_cb_packet,
    std::vector<CoreCoord> const& worker_cores,
    std::unordered_map<CoreCoord, size_t>& math_page_counts_out) {
    for (size_t w = 0; w < worker_slices.size(); w++) {
        auto const& w_logical = worker_cores[w];
        auto const& slices = worker_slices[w];
        math_page_counts_out[w_logical] = compute_math_pages_from_tensor_slices(slices, pages_per_cb_packet);
    }
}

// More efficient implementation is to do the splitting outside but we'll do that after we have something working
// Outer index is per worker, inner is each command stream (0 and 1 respectively for that worker)
// second result is total number of pages cycled through the CBs
static void generate_partial_reducer_reader_worker_command_streams(
    ReduceScatterBuilderConfig& builder_config,
    std::optional<TensorSyncSpec> const& in0_tensor_sync,
    std::optional<TensorSyncSpec> const& in1_tensor_sync,
    // Same for both operands
    std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> const& worker_tensor_slices,
    std::vector<CoreCoord> const& worker_cores,
    WorkerCommandStreams& worker_command_streams_out) {
    using namespace ttnn::ccl::cmd;
    using namespace ttnn::ccl::cmd::uops;
    using namespace ttnn::ccl::cmd::builder;

    auto const& reader_cbs = builder_config.all_cbs.get().partial_reducer_reader;
    auto const& topology_config = builder_config.topology_config.get();

    const size_t num_workers = worker_cores.size();
    log_trace(
        tt::LogOp, "generate_partial_reducer_reader_worker_command_streams. topologyu: {}", topology_config.topology());

    bool in0_async_mode_specified = in0_tensor_sync.has_value();
    bool in1_async_mode_specified = in1_tensor_sync.has_value();
    TT_FATAL(in1_async_mode_specified, "Internal error. Expected input tensor sync to be populated");
    auto const& from_remote_input_tensor_sync = in1_tensor_sync;
    TT_FATAL(
        worker_tensor_slices.size() == num_workers,
        "Internal error. Expected number of worker slices to match number of workers");
    auto get_cb_base = [](size_t slice_index, ttnn::ccl::Topology topology, uint32_t idx0_cb, uint32_t default_cb) {
        if (topology == ttnn::ccl::Topology::Linear) {
            return default_cb;
        } else {
            return slice_index == 0 ? idx0_cb : default_cb;
        }
    };
    auto get_cb = std::bind(
        get_cb_base, std::placeholders::_1, topology_config.topology(), reader_cbs.pass_through, reader_cbs.math_in0);

    for (size_t w = 0; w < num_workers; w++) {
        auto const& w_logical = worker_cores[w];
        {
            auto& worker_command_stream0 = worker_command_streams_out.reader_cmds0[w_logical];
            for (size_t i = 0; i < worker_tensor_slices[w].size(); i++) {
                bool last_slice = i == worker_tensor_slices[w].size() - 1;
                auto const& s = worker_tensor_slices[w][i];
                if (in0_tensor_sync.has_value()) {
                    // NOTE: per-worker sync
                    worker_command_stream0.push_back(
                        local_semaphore_wait(in0_tensor_sync.value().get_tensor_sync_semaphore(w), i + 1));
                }
                if (last_slice) {
                    // Make sure not to add the space at the beginning of the CB chunk for packet header
                    // so when we write out from the other side, we maintain proper alignment
                    worker_command_stream0.push_back(read_tensor_slice_to_cb(s, get_cb(i)));
                } else {
                    worker_command_stream0.push_back(read_tensor_slice_to_cb_for_eventual_fabric_write(s, get_cb(i)));
                }
            }
        }
        {
            auto& worker_command_stream1 = worker_command_streams_out.reader_cmds1[w_logical];
            for (size_t i = 0; i < worker_tensor_slices[w].size(); i++) {
                bool last_slice = i == worker_tensor_slices[w].size() - 1;
                auto const& s = worker_tensor_slices[w][i];
                worker_command_stream1.push_back(
                    local_semaphore_wait(from_remote_input_tensor_sync.value().get_tensor_sync_semaphore(w), i + 1));
                if (last_slice) {
                    worker_command_stream1.push_back(read_tensor_slice_to_cb(s, reader_cbs.math_in1));
                } else {
                    worker_command_stream1.push_back(
                        read_tensor_slice_to_cb_for_eventual_fabric_write(s, reader_cbs.math_in1));
                }
            }
        }
    }
}

static void generate_partial_reducer_writer_worker_command_streams(
    ReduceScatterBuilderConfig& builder_config,
    TensorSyncBundle const& remote_output_tensor_sync_bundle,
    TensorSyncBundle const& local_partial_output_tensor_sync_bundle,
    std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> const& remote_out_worker_tensor_slices,
    LineDirection direction,
    WorkerCommandStreams& worker_command_streams) {
    auto const& topology_config = builder_config.topology_config.get();
    auto const& worker_cores = builder_config.worker_cores.get().partial_reducers[direction];
    auto const& worker_cores_vec = builder_config.worker_cores.get().partial_reducers_vec[direction];
    size_t num_devices = topology_config.line_size();
    bool is_forward_direction = direction == LineDirection::FORWARD;

    log_trace(
        tt::LogOp,
        "generate_partial_reducer_writer_worker_command_streams. topologyu: {}, num_devices: {}",
        topology_config.topology(),
        num_devices);

    using namespace ttnn::ccl::cmd;
    using namespace ttnn::ccl::cmd::uops;
    using namespace ttnn::ccl::cmd::builder;

    auto const& writer_cbs = builder_config.all_cbs.get().partial_reducer_writer;
    TT_FATAL(
        local_partial_output_tensor_sync_bundle.sync_spec.has_value(),
        "Internal error. Expected local partial output tensor to have synchronization info");
    // Since Command processor currently doesn't support switching between tensors within a single command stream
    // (future work), we split into two command streams, with each one assigned to one of the two output tensors:
    //  0. Remote output tensor
    //  1. Local output tensor
    //
    // After all slices have been forwarded to the remote chip, then the command streams synchronize with each other
    // to indicate that the "from math" CB can be read from

    const size_t num_workers = worker_cores.num_cores();

    auto const local_partial_output_tensor_slice =
        convert_to_whole_tensor_slice(*local_partial_output_tensor_sync_bundle.tensor);
    auto const local_output_tensor_slices_per_worker =
        split_tensor_slices_across_workers_page_aligned(num_workers, {local_partial_output_tensor_slice});
    TT_FATAL(
        local_output_tensor_slices_per_worker.size() == num_workers,
        "Local output tensor slices per worker size mismatch");
    TT_FATAL(
        remote_out_worker_tensor_slices.size() == num_workers, "Remote output tensor slices per worker size mismatch");

    auto get_cb_base = [](size_t slice_index, ttnn::ccl::Topology topology, uint32_t idx0_cb, uint32_t default_cb) {
        if (topology == ttnn::ccl::Topology::Linear) {
            return default_cb;
        } else {
            return slice_index == 0 ? idx0_cb : default_cb;
        }
    };
    log_trace(
        tt::LogOp,
        "\t\t\twriter_cbs.pass_through: {}, writer_cbs.math_out: {}",
        writer_cbs.pass_through,
        writer_cbs.math_out);
    auto get_cb = std::bind(
        get_cb_base, std::placeholders::_1, topology_config.topology(), writer_cbs.pass_through, writer_cbs.math_out);

    TT_FATAL(
        remote_output_tensor_sync_bundle.sync_spec.has_value(),
        "Internal error. Expected remote output tensor to have synchronization info");
    auto const& remote_out_tensor_sync = remote_output_tensor_sync_bundle.sync_spec.value();

    std::vector<std::vector<CclHostLowLevelWorkerCommand>> writer_command_streams_per_worker;
    auto const next_chip_fabric_unicast = UnicastCommandDestArgs{1, is_forward_direction};
    auto internal_command_stream_sync_sem_id = CreateSemaphore(builder_config.program.get(), worker_cores, 0);
    for (size_t w = 0; w < num_workers; w++) {
        // command_streams.push_back({});
        {  // Command stream 0
            const size_t operand_index = 0;
            auto& worker_command_stream0 = worker_command_streams.writer_cmds0[worker_cores_vec[w]];
            for (size_t i = 0; i < remote_out_worker_tensor_slices[w].size(); i++) {
                auto const& s = remote_out_worker_tensor_slices[w][i];
                log_debug(
                    tt::LogOp,
                    "Worker {} Writer Kernel cmds0[{}]: tensor_slice: (.shape=(w={},z={},y={},x={}), "
                    ".slice_shape=(w={},z={},y={},x={})), .slice_offset=(w={},z={},y={},x={}), "
                    ".worker_slice_shape=(w={},z={},y={},x={}), .worker_slice_offset=(w={},z={},y={},x={}), cb_id={}",
                    w,
                    2 * i,
                    s.tensor_slice_shape.w,
                    s.tensor_slice_shape.z,
                    s.tensor_slice_shape.y,
                    s.tensor_slice_shape.x,
                    s.tensor_slice_shape.w,
                    s.tensor_slice_shape.z,
                    s.tensor_slice_shape.y,
                    s.tensor_slice_shape.x,
                    s.tensor_slice_offset.w,
                    s.tensor_slice_offset.z,
                    s.tensor_slice_offset.y,
                    s.tensor_slice_offset.x,
                    s.worker_slice_shape.w,
                    s.worker_slice_shape.z,
                    s.worker_slice_shape.y,
                    s.worker_slice_shape.x,
                    s.worker_slice_offset.w,
                    s.worker_slice_offset.z,
                    s.worker_slice_offset.y,
                    s.worker_slice_offset.x,
                    get_cb(i));

                worker_command_stream0.push_back(
                    fabric_write_cb_to_tensor_slice(s, get_cb(i), next_chip_fabric_unicast));

                // remote_out_tensor_sync
                worker_command_stream0.push_back(fabric_unicast_semaphore_inc_mcast(
                    // For now we assume the semaphores are consistent across chips
                    // though this may not be generally safe - it should be for the initial
                    // experimental cases we care about
                    // NOTE: per worker semaphore
                    remote_out_tensor_sync.get_tensor_sync_semaphore(w),
                    CclCommandAtomicInc{1},
                    remote_out_tensor_sync.get_target(w),
                    next_chip_fabric_unicast)

                );
            }
            // Finish off by notifying the other command stream that it's safe for it to pull from the
            // "from math" CB
            worker_command_stream0.push_back(local_core_semaphore_inc(internal_command_stream_sync_sem_id, 1));
        }
        {  // Command stream 1
            const size_t operand_index = 1;
            auto& worker_command_stream1 = worker_command_streams.writer_cmds1[worker_cores_vec[w]];

            TT_FATAL(
                local_output_tensor_slices_per_worker[w].size() == 1,
                "Local output tensor expected only to have a single tensor slice");
            // Wait for all-clear from first command stream that "from math" CB is no longer being pulled from
            // Then it's safe to forward to fabric from CB

            std::ranges::copy(
                CclHostLowLevelCommandSequence{
                    local_semaphore_wait(internal_command_stream_sync_sem_id, 1),
                    local_write_cb_to_tensor_slice(local_output_tensor_slices_per_worker[w][0], writer_cbs.math_out),
                    local_chip_semaphore_inc_mcast(
                        // NOTE: Per worker semaphores
                        local_partial_output_tensor_sync_bundle.sync_spec.value().get_tensor_sync_semaphore(w),
                        CclCommandAtomicInc{1},
                        local_partial_output_tensor_sync_bundle.sync_spec.value().get_target(w))},
                std::back_inserter(worker_command_stream1));
        }
    }
}

// TODO: optimize to have set block_size == packet_size
static std::vector<uint32_t> generate_reduce_op_kernel_rt_args(size_t total_num_math_pages) {
    auto const& args = std::vector<uint32_t>{total_num_math_pages, 1};

    std::size_t i = 0;
    log_trace(tt::LogOp, "\tReduce Scatter Worker RT Args:");
    log_trace(tt::LogOp, "\t\tblock_size: {}", args.at(i++));
    log_trace(tt::LogOp, "\t\ttotal_num_math_pages: {}", args.at(i++));
    TT_ASSERT(args.size() == i, "Missed some args");

    return args;
}

static void set_math_runtime_args(
    Program& program, KernelHandle math_kernel_id, CoreCoord const& worker_logical, size_t total_num_math_pages) {
    log_trace(tt::LogOp, "Setting math kernel RT args");
    auto rt_args = generate_reduce_op_kernel_rt_args(total_num_math_pages);
    tt::tt_metal::SetRuntimeArgs(program, math_kernel_id, worker_logical, rt_args);
}


static void create_non_end_of_line_final_reducer_worker_commands(
    ReduceScatterBuilderConfig& builder_config,
    WorkerCommandStreams& worker_command_streams_out,
    std::unordered_map<CoreCoord, size_t>& math_page_counts_out) {
    auto const& final_reducer_worker_cores = builder_config.worker_cores.get().final_reducers_vec;
    auto const& all_program_tensors = builder_config.all_tensors.get();
    auto const& all_cbs = builder_config.all_cbs.get();
    log_trace(tt::LogOp, "--------------------------------------");
    log_trace(tt::LogOp, "CREATE WORKER (final reducer - not end. Device={})", builder_config.device->id());

    size_t const num_partial_reducer_workers_per_direction =
        builder_config.worker_cores.get().partial_reducers[LineDirection::FORWARD].size();

    std::array<TensorSyncBundle, 2> const& partial_output_tensor_sync_bundles = {
        TensorSyncBundle{
            all_program_tensors.local_output_partial[LineDirection::FORWARD],
            all_program_tensors.local_output_partial_sync[LineDirection::FORWARD]},
        TensorSyncBundle{
            all_program_tensors.local_output_partial[LineDirection::BACKWARD],
            all_program_tensors.local_output_partial_sync[LineDirection::BACKWARD]},
    };

    generate_final_reducer_reader_worker_command_streams(
        builder_config,
        partial_output_tensor_sync_bundles[LineDirection::FORWARD],
        partial_output_tensor_sync_bundles[LineDirection::BACKWARD],
        worker_command_streams_out,
        math_page_counts_out);

    generate_final_reducer_writer_worker_command_streams(
        builder_config,
        TensorSyncBundle{all_program_tensors.local_output_tensor, all_program_tensors.local_output_sync},
        worker_command_streams_out);

    TT_FATAL(final_reducer_worker_cores.size() > 0, "Internal error. No final reducer cores were created");
}

static void populate_partial_reduce_worker_commands(
    ReduceScatterBuilderConfig& builder_config,

    std::array<std::vector<std::vector<ttnn::ccl::v2::TensorSlice>>, 2> const& reader_worker_slices_by_direction,
    std::array<std::vector<std::vector<ttnn::ccl::v2::TensorSlice>>, 2> const& writer_worker_slices_by_direction,

    WorkerCommandStreams& worker_command_streams_out,
    std::unordered_map<CoreCoord, size_t>& math_page_counts_out) {
    auto const& partial_reducer_worker_cores = builder_config.worker_cores.get().partial_reducers_vec;
    auto const& all_tensors = builder_config.all_tensors.get();
    auto const& all_cbs = builder_config.all_cbs.get();
    auto const& topology_config = builder_config.topology_config.get();
    auto const& kernel_ids = builder_config.kernel_ids.get();
    log_trace(tt::LogOp, "--------------------------------------");
    log_trace(tt::LogOp, "CREATE WORKER (partial reducer - not end. Device={})", builder_config.device->id());

    std::array<std::vector<CoreCoord>, 2> partial_reducer_worker_cores_vec = {
        partial_reducer_worker_cores[LineDirection::FORWARD], partial_reducer_worker_cores[LineDirection::BACKWARD]};
    compute_math_pages_from_per_worker_tensor_slices(
        reader_worker_slices_by_direction[LineDirection::FORWARD],
        builder_config.pages_per_cb_packet,
        partial_reducer_worker_cores_vec[LineDirection::FORWARD],
        math_page_counts_out);
    compute_math_pages_from_per_worker_tensor_slices(
        reader_worker_slices_by_direction[LineDirection::BACKWARD],
        builder_config.pages_per_cb_packet,
        partial_reducer_worker_cores_vec[LineDirection::BACKWARD],
        math_page_counts_out);

    for (auto line_direction : {LineDirection::FORWARD, LineDirection::BACKWARD}) {
        // Logic for any chip in the "middle" of the line
        bool is_forward_direction = line_direction == LineDirection::FORWARD;
        generate_partial_reducer_reader_worker_command_streams(
            builder_config,
            all_tensors.input_tensor_sync,
            all_tensors.input_tensor_from_remote_sync[line_direction],
            reader_worker_slices_by_direction[line_direction],
            partial_reducer_worker_cores_vec[line_direction],
            worker_command_streams_out);

        generate_partial_reducer_writer_worker_command_streams(
            builder_config,
            TensorSyncBundle{all_tensors.remote_output[line_direction], all_tensors.remote_output_sync[line_direction]},
            TensorSyncBundle{
                all_tensors.local_output_partial[line_direction],
                all_tensors.local_output_partial_sync[line_direction]},
            writer_worker_slices_by_direction[line_direction],
            line_direction,
            worker_command_streams_out);
    }
}

static void create_final_reducer_worker_rt_args_not_end_of_line(
    ReduceScatterBuilderConfig& builder_config,
    fabric_lifetime_mode fabric_mode,
    WorkerCommandStreams& worker_command_streams_out,
    std::unordered_map<CoreCoord, size_t>& math_page_counts_out) {
    using namespace ttnn::ccl::worker_detail;

    auto const& final_reducer_worker_cores = builder_config.worker_cores.get().final_reducers_vec;
    auto const& all_program_tensors = builder_config.all_tensors.get();

    for (size_t i = 0; i < final_reducer_worker_cores.size(); i++) {
        auto const& w_logical = final_reducer_worker_cores[i];
        generate_multi_input_command_stream_kernel_rt_args(
            builder_config.program,
            builder_config.kernel_ids.get().reader,
            {all_program_tensors.local_output_partial[LineDirection::FORWARD],
             all_program_tensors.local_output_partial[LineDirection::BACKWARD]},
            {builder_config.page_size, builder_config.page_size},
            builder_config.device,
            builder_config.pages_per_cb_packet,
            {w_logical},
            worker_command_streams_out.reader_cmds0.at(w_logical),
            worker_command_streams_out.reader_cmds1.at(w_logical),
            std::nullopt,
            std::nullopt);
        set_math_runtime_args(
            builder_config.program,
            builder_config.kernel_ids.get().math,
            w_logical,
            math_page_counts_out.at(w_logical));
        generate_multi_input_command_stream_kernel_rt_args(
            builder_config.program,
            builder_config.kernel_ids.get().writer,
            {all_program_tensors.local_output_tensor, nullptr},
            {builder_config.page_size, builder_config.page_size},
            builder_config.device,
            builder_config.pages_per_cb_packet,
            {w_logical},
            worker_command_streams_out.writer_cmds0.at(w_logical),
            ttnn::ccl::cmd::CclHostLowLevelCommandSequence{},
            std::nullopt,
            std::nullopt);
    }
}

static void populate_partial_reduce_rt_args(
    ReduceScatterBuilderConfig& builder_config,

    WorkerCommandStreams& worker_command_streams_out,
    std::unordered_map<CoreCoord, size_t>& math_page_counts_out) {
    using namespace ttnn::ccl::worker_detail;
    using Direction = ttnn::ccl::EdmLineFabricOpInterface::Direction;

    auto& fabric = builder_config.fabric.get();
    auto const& all_tensors = builder_config.all_tensors.get();
    auto const& kernel_ids = builder_config.kernel_ids.get();
    auto device = builder_config.device;

    auto get_fabric_connection = [&device, &fabric](bool is_connected, Direction dir) {
        return is_connected
                   ? std::make_optional<ttnn::ccl::SenderWorkerAdapterSpec>(fabric.uniquely_connect_worker(device, dir))
                   : std::nullopt;
    };

    auto const& partial_reducer_worker_cores = builder_config.worker_cores.get().partial_reducers_vec;
    std::array<std::vector<CoreCoord>, 2> partial_reducer_worker_cores_vec = {
        partial_reducer_worker_cores[LineDirection::FORWARD], partial_reducer_worker_cores[LineDirection::BACKWARD]};

    for (auto line_direction : {LineDirection::FORWARD, LineDirection::BACKWARD}) {
        bool is_forward_direction = line_direction == LineDirection::FORWARD;
        auto fwd_fabric_connection = get_fabric_connection(is_forward_direction, Direction::FORWARD);
        auto bwd_fabric_connection = get_fabric_connection(!is_forward_direction, Direction::BACKWARD);

        for (size_t i = 0; i < partial_reducer_worker_cores_vec[line_direction].size(); i++) {
            auto const& w_logical = partial_reducer_worker_cores_vec[line_direction][i];
            // Reader kernel RT args
            generate_multi_input_command_stream_kernel_rt_args(
                builder_config.program.get(),
                kernel_ids.reader,
                std::vector<Tensor const*>{
                    all_tensors.input_tensor, all_tensors.input_tensor_from_remote[line_direction]},
                {builder_config.page_size, builder_config.page_size},
                builder_config.device,
                builder_config.pages_per_cb_packet,  // TODO: get from fabric
                {w_logical},
                worker_command_streams_out.reader_cmds0.at(w_logical),
                worker_command_streams_out.reader_cmds1.at(w_logical),
                std::nullopt,
                std::nullopt);
            set_math_runtime_args(
                builder_config.program.get(), kernel_ids.math, w_logical, math_page_counts_out[w_logical]);
            auto output_tensor_ptrs = std::vector<Tensor const*>{
                all_tensors.remote_output[line_direction], all_tensors.local_output_partial[line_direction]};
            generate_multi_input_command_stream_kernel_rt_args(
                builder_config.program.get(),
                kernel_ids.writer,
                output_tensor_ptrs,
                {builder_config.page_size, builder_config.page_size},
                builder_config.device,
                builder_config.pages_per_cb_packet,  // TODO: get from fabric
                {w_logical},
                worker_command_streams_out.writer_cmds0.at(w_logical),
                worker_command_streams_out.writer_cmds1.at(w_logical),
                fwd_fabric_connection,
                bwd_fabric_connection,
                std::unordered_map<const Tensor*, Device*>{
                    {output_tensor_ptrs[0],
                     line_direction == LineDirection::FORWARD ? builder_config.forward_device
                                                              : builder_config.backward_device}});
        }
        //////////////
    }
}

static void create_worker_runtime_args_for_inactive_workers(ReduceScatterBuilderConfig& builder_config) {
    auto const& inactive_cores = builder_config.worker_cores.get().final_reducers;
    using namespace ttnn::ccl::worker_detail;
    log_trace(tt::LogOp, "--------------------------------------");
    log_trace(tt::LogOp, "CREATE WORKER (inactive - not end. Device={})", builder_config.device->id());

    generate_multi_input_command_stream_kernel_rt_args(
        builder_config.program.get(),
        builder_config.kernel_ids.get().reader,
        {nullptr, nullptr},
        {0, 0},
        builder_config.device,
        0,  // TODO: get from fabric
        inactive_cores,
        ttnn::ccl::cmd::CclHostLowLevelCommandSequence{},
        ttnn::ccl::cmd::CclHostLowLevelCommandSequence{},
        std::nullopt,
        std::nullopt);

    tt::tt_metal::SetRuntimeArgs(
        builder_config.program.get(),
        builder_config.kernel_ids.get().math,
        inactive_cores,
        generate_reduce_op_kernel_rt_args(0));

    generate_multi_input_command_stream_kernel_rt_args(
        builder_config.program.get(),
        builder_config.kernel_ids.get().writer,
        {nullptr, nullptr},
        {0, 0},
        builder_config.device,
        0,  // TODO: get from fabric
        inactive_cores,
        ttnn::ccl::cmd::CclHostLowLevelCommandSequence{},
        ttnn::ccl::cmd::CclHostLowLevelCommandSequence{},
        std::nullopt,
        std::nullopt);
}

static void validate_end_of_line_worker_tensors(
    ReduceScatterBuilderConfig& builder_config, fabric_lifetime_mode fabric_mode) {
    ProgramTensorsBundle const& all_tensors = builder_config.all_tensors.get();
    LineTopology const& line_topology = builder_config.topology_config.get();
    bool teardown_fabric = fabric_mode == fabric_lifetime_mode::TRANSIENT;

    TT_FATAL(all_tensors.input_tensor != nullptr, "Input tensor must be populated");
    TT_FATAL(all_tensors.local_output_tensor != nullptr, "Output tensor must be populated");
    if (line_topology.is_first_device_in_line(LineDirection::FORWARD)) {
        TT_FATAL(
            all_tensors.input_tensor_from_remote[LineDirection::FORWARD] == nullptr,
            "Input tensor from remote must be populated");
        TT_FATAL(
            all_tensors.input_tensor_from_remote[LineDirection::BACKWARD] != nullptr,
            "Input tensor from remote must be populated");
        TT_FATAL(
            all_tensors.input_tensor->shape() == all_tensors.input_tensor_from_remote[LineDirection::BACKWARD]->shape(),
            "Input tensor and input from remote tensor must have the same shape");
    }
    if (line_topology.is_first_device_in_line(LineDirection::BACKWARD)) {
        TT_FATAL(
            all_tensors.input_tensor_from_remote[LineDirection::BACKWARD] == nullptr,
            "Input tensor from remote must be populated");
        TT_FATAL(
            all_tensors.input_tensor_from_remote[LineDirection::FORWARD] != nullptr,
            "Input tensor from remote must be populated");
        TT_FATAL(
            all_tensors.input_tensor->shape() == all_tensors.input_tensor_from_remote[LineDirection::FORWARD]->shape(),
            "Input tensor and input from remote tensor must have the same shape");
    }
}

static void create_end_of_line_worker_commands(
    ReduceScatterBuilderConfig& builder_config,
    std::unordered_map<CoreCoord, size_t>& worker_math_page_counts_out,
    WorkerCommandStreams& worker_command_streams_out) {
    // using namespace ttnn::ccl::worker_detail;
    using namespace ttnn::ccl::cmd;
    using namespace ttnn::ccl::cmd::uops;
    using namespace ttnn::ccl::cmd::builder;
    auto const& topology_config = builder_config.topology_config.get();
    auto const& worker_cores = builder_config.worker_cores.get();
    auto const& all_tensors = builder_config.all_tensors.get();
    auto const& all_cbs = builder_config.all_cbs.get();

    size_t nchips = builder_config.topology_config.get().line_size();
    size_t curr_chip = builder_config.topology_config.get().line_index();
    auto num_workers = worker_cores.partial_reducers_vec[LineDirection::FORWARD].size();

    TT_FATAL(
        worker_cores.partial_reducers_vec[LineDirection::BACKWARD].size() == num_workers,
        "Internal error. Expected number of workers to match");
    // out_slices = partial_out_tensor.chunk(n=line_size,dim=dim)
    // out_slices_fwd = reverse(out_slices[line_topology.line_index() + 1:])
    // worker_out_slices_fwd = distribute_across_workers(out_slices_fwd)
    // out_slices_bwd = out_slices[:line_topology.line_index() + 1] // assuming exclusive end
    // worker_out_slices_bwd = distribute_across_workers(out_slices_bwd, n_workers)
    auto const reader_in_slices =
        generate_tensor_slices(nchips, *builder_config.all_tensors.get().input_tensor, builder_config.dim);

    auto reader_slices_fwd =
        vslice(reader_in_slices, reader_in_slices.size() - 1, std::min(curr_chip + 1, reader_in_slices.size() - 1));
    auto reader_slices_bwd =
        vslice(reader_in_slices, 0, curr_chip - !topology_config.is_first_device_in_line(LineDirection::FORWARD));
    auto remote_writer_slices_fwd =
        vslice(reader_in_slices, reader_in_slices.size() - 1, std::min(curr_chip + 1, reader_in_slices.size() - 1));
    auto remote_writer_slices_bwd =
        vslice(reader_in_slices, 0, curr_chip - !topology_config.is_first_device_in_line(LineDirection::FORWARD));

    auto reader_worker_sliced_fwd = split_tensor_slices_across_workers_page_aligned(num_workers, reader_slices_fwd);
    auto reader_worker_sliced_bwd = split_tensor_slices_across_workers_page_aligned(num_workers, reader_slices_bwd);
    auto remote_writer_worker_sliced_fwd =
        split_tensor_slices_across_workers_page_aligned(num_workers, remote_writer_slices_fwd);
    auto remote_writer_worker_sliced_bwd =
        split_tensor_slices_across_workers_page_aligned(num_workers, remote_writer_slices_bwd);

    std::array<decltype(reader_worker_sliced_fwd), 2> reader_worker_slices = {
        reader_worker_sliced_fwd, reader_worker_sliced_bwd};
    std::array<decltype(remote_writer_worker_sliced_fwd), 2> remote_writer_worker_slices = {
        remote_writer_worker_sliced_fwd, remote_writer_worker_sliced_bwd};

    std::array<std::vector<CoreCoord>, 2> const reader_worker_cores_per_direction = worker_cores.partial_reducers_vec;
    std::array<std::vector<CoreCoord>, 2> const& writer_worker_cores_per_direction = reader_worker_cores_per_direction;

    auto const local_partial_output_tensor_slice = convert_to_whole_tensor_slice(*all_tensors.local_output_tensor);
    auto writer_end_of_line_output_worker_slices =
        split_tensor_slices_across_workers_page_aligned(num_workers, {local_partial_output_tensor_slice});
    TT_FATAL(
        writer_end_of_line_output_worker_slices.size() == num_workers,
        "Internal error. Expected number of end of line worker slices to match number of workers. Got {} but expected "
        "{}",
        writer_end_of_line_output_worker_slices.size(),
        num_workers);

    for (auto direction : {LineDirection::FORWARD, LineDirection::BACKWARD}) {
        bool is_forward_direction = direction == LineDirection::FORWARD;
        bool is_start_of_line = topology_config.is_first_device_in_line(direction);

        auto const& reader_worker_cores = reader_worker_cores_per_direction[direction];
        TT_FATAL(
            reader_worker_cores.size() == num_workers,
            "Internal error. Expected number of reader worker cores to match number of workers. Got {} but expected {}",
            reader_worker_cores.size(),
            num_workers);

        std::vector<std::vector<CclHostLowLevelWorkerCommand>> worker_in0_cmd_stream(num_workers);
        std::optional<std::vector<std::vector<CclHostLowLevelWorkerCommand>>> worker_in1_cmd_stream;
        std::vector<std::vector<CclHostLowLevelWorkerCommand>> worker_out0_cmd_stream(num_workers);
        TT_FATAL(
            reader_worker_slices[direction].size() == num_workers,
            "Internal error. Expected number of reader worker slices to match number of workers. Got {} but expected "
            "{}",
            reader_worker_slices[direction].size(),
            num_workers);
        TT_FATAL(
            reader_worker_slices[direction].size() == num_workers,
            "Internal error. Expected number of writer worker slices to match number of workers. Got {} but expected "
            "{}",
            reader_worker_slices[direction].size(),
            num_workers);
        if (!is_start_of_line) {
            worker_in1_cmd_stream = std::vector<std::vector<CclHostLowLevelWorkerCommand>>(num_workers);
        }
        for (size_t i = 0; i < num_workers; i++) {
            auto const& w_logical = reader_worker_cores[i];
            auto& in0_cmd_stream = worker_command_streams_out.reader_cmds0[w_logical];   // worker_in0_cmd_stream[i];
            auto& out0_cmd_stream = worker_command_streams_out.writer_cmds0[w_logical];  // worker_out0_cmd_stream[i];
            auto& in1_cmd_stream = worker_command_streams_out.reader_cmds1[w_logical];

            size_t num_math_pages = 0;
            if (is_start_of_line) {
                for (auto const& slice : reader_worker_slices[direction][i]) {
                    in0_cmd_stream.push_back(read_tensor_slice_to_cb_for_eventual_fabric_write(
                        slice, all_cbs.line_start_reader.pass_through));
                }

                for (size_t s = 0; s < remote_writer_worker_slices[direction][i].size(); s++) {
                    auto const& slice = remote_writer_worker_slices[direction][i][s];
                    out0_cmd_stream.push_back(fabric_write_cb_to_tensor_slice(
                        slice,
                        all_cbs.line_start_writer.pass_through,
                        UnicastCommandDestArgs{1, direction == LineDirection::FORWARD}));
                    out0_cmd_stream.push_back(fabric_unicast_semaphore_inc_mcast(
                        // NOTE: per worker semaphores
                        all_tensors.remote_output_sync.at(direction).get_tensor_sync_semaphore(i),
                        CclCommandAtomicInc{1},
                        all_tensors.remote_output_sync.at(direction).get_target(i),
                        UnicastCommandDestArgs{1, direction == LineDirection::FORWARD}));
                }
            } else {
                auto const& worker_in_slices = reader_worker_slices.at(direction).at(i);
                // READER COMMANDS
                auto const& from_remote_sync = direction == LineDirection::FORWARD
                                                   ? all_tensors.input_tensor_from_remote_sync[LineDirection::FORWARD]
                                                   : all_tensors.input_tensor_from_remote_sync[LineDirection::BACKWARD];
                TT_FATAL(worker_in_slices.size() == 1, "Internal error. Expected only one slice per worker");
                in0_cmd_stream.push_back(
                    read_tensor_slice_to_cb(worker_in_slices[0], all_cbs.line_end_reader.math_in0));
                // NOTE: per worker semaphore
                in1_cmd_stream.push_back(local_semaphore_wait(from_remote_sync.get_tensor_sync_semaphore(0), 1));
                in1_cmd_stream.push_back(
                    read_tensor_slice_to_cb(worker_in_slices.at(0), all_cbs.line_end_reader.math_in1));

                // MATH PAGE COUNTS
                num_math_pages =
                    compute_math_pages_from_tensor_slices(worker_in_slices, builder_config.pages_per_cb_packet);

                // WRITER COMMANDS
                TT_FATAL(
                    writer_end_of_line_output_worker_slices[i].size() == 1,
                    "Internal error. Expected only one slice per worker");
                out0_cmd_stream.push_back(local_write_cb_to_tensor_slice(
                    writer_end_of_line_output_worker_slices[i][0], all_cbs.line_end_writer.math_out));
            }

            worker_math_page_counts_out[w_logical] = num_math_pages;
        }
    }
}

// Maybe reusable for all configurations
static void create_end_of_line_worker_runtime_args(
    ReduceScatterBuilderConfig& builder_config,
    WorkerCommandStreams& worker_command_streams,
    std::unordered_map<CoreCoord, size_t> const& worker_math_page_counts) {
    using namespace ttnn::ccl::worker_detail;
    using namespace ttnn::ccl::cmd;
    using Direction = ttnn::ccl::EdmLineFabricOpInterface::Direction;
    Program& program = builder_config.program.get();
    Device* device = builder_config.device;
    ttnn::ccl::EdmLineFabricOpInterface& fabric = builder_config.fabric.get();
    ProgramTensorsBundle const& all_tensors = builder_config.all_tensors.get();
    ReduceScatterKernelHandles const& kernel_ids = builder_config.kernel_ids.get();
    WorkerCoreBundle const& worker_cores = builder_config.worker_cores.get();

    auto get_fabric_connection = [&device, &fabric](bool is_connected, Direction dir) {
        return is_connected
                   ? std::make_optional<ttnn::ccl::SenderWorkerAdapterSpec>(fabric.uniquely_connect_worker(device, dir))
                   : std::nullopt;
    };

    std::array<std::vector<CoreCoord>, 2> const reader_worker_cores_per_direction = worker_cores.partial_reducers_vec;
    std::array<std::vector<CoreCoord>, 2> const& writer_worker_cores_per_direction = reader_worker_cores_per_direction;
    auto num_workers = worker_cores.partial_reducers_vec[LineDirection::FORWARD].size();

    // Generate the kernels themselves
    for (auto direction : {LineDirection::FORWARD, LineDirection::BACKWARD}) {
        bool is_start_of_line = builder_config.topology_config.get().is_first_device_in_line(direction);
        auto const& reader_worker_cores = reader_worker_cores_per_direction[direction];
        bool is_forward_direction = direction == LineDirection::FORWARD;

        auto fwd_fabric_connection =
            get_fabric_connection(is_forward_direction && is_start_of_line, Direction::FORWARD);
        auto bwd_fabric_connection =
            get_fabric_connection(!is_forward_direction && is_start_of_line, Direction::BACKWARD);

        Tensor* output_tensor_ptr = nullptr;
        auto input_tensor_ptrs = std::vector<Tensor const*>{nullptr, nullptr};
        input_tensor_ptrs[0] = all_tensors.input_tensor;

        if (is_start_of_line) {
            output_tensor_ptr = all_tensors.remote_output[direction];
        } else {
            output_tensor_ptr = all_tensors.local_output_tensor;
            input_tensor_ptrs[1] = all_tensors.input_tensor_from_remote.at(direction);
            TT_FATAL(input_tensor_ptrs[1] != nullptr, "Internal error. Expected input tensor to be populated");
        }

        for (size_t i = 0; i < num_workers; i++) {
            auto const& w_logical = reader_worker_cores[i];
            size_t num_math_pages = is_start_of_line ? 0 : worker_math_page_counts.at(w_logical);

            TT_FATAL(output_tensor_ptr != nullptr, "Internal error. Expected output tensor to be populated");
            TT_FATAL(input_tensor_ptrs[0] != nullptr, "Internal error. Expected input tensor to be populated");
            TT_FATAL(
                worker_command_streams.reader_cmds0.find(w_logical) != worker_command_streams.reader_cmds0.end(),
                "Internal error. Expected reader command stream to be populated");
            bool has_in1_commands =
                worker_command_streams.reader_cmds1.find(w_logical) != worker_command_streams.reader_cmds1.end();
            generate_multi_input_command_stream_kernel_rt_args(
                program,
                kernel_ids.reader,
                input_tensor_ptrs,
                {builder_config.page_size, builder_config.page_size},
                device,
                builder_config.pages_per_cb_packet,
                {w_logical},
                worker_command_streams.reader_cmds0.at(w_logical),
                has_in1_commands ? worker_command_streams.reader_cmds1.at(w_logical)
                                 : std::vector<CclHostLowLevelWorkerCommand>{},
                std::nullopt,
                std::nullopt);
            set_math_runtime_args(program, kernel_ids.math, w_logical, num_math_pages);
            generate_multi_input_command_stream_kernel_rt_args(
                program,
                kernel_ids.writer,
                {output_tensor_ptr, nullptr},
                {builder_config.page_size, builder_config.page_size},
                device,
                builder_config.pages_per_cb_packet,
                {w_logical},
                worker_command_streams.writer_cmds0.at(w_logical),
                std::vector<CclHostLowLevelWorkerCommand>{},
                fwd_fabric_connection,
                bwd_fabric_connection);
        }
    }
}



static void create_end_of_line_worker_commands(
    ReduceScatterBuilderConfig& builder_config,
    fabric_lifetime_mode fabric_mode,
    WorkerCommandStreams& worker_command_streams,
    std::unordered_map<CoreCoord, size_t>& worker_math_page_counts) {
    using namespace ttnn::ccl::worker_detail;
    using namespace ttnn::ccl::cmd;
    using namespace ttnn::ccl::cmd::uops;
    using namespace ttnn::ccl::cmd::builder;

    validate_end_of_line_worker_tensors(builder_config, fabric_mode);

    log_trace(tt::LogOp, "--------------------------------------");
    log_trace(tt::LogOp, "CREATE WORKER (end of line Device={})", builder_config.device->id());

    create_end_of_line_worker_commands(builder_config, worker_math_page_counts, worker_command_streams);
}

static void validate_non_end_of_line_tensors(ReduceScatterBuilderConfig& builder_config) {
    auto const& all_program_tensors = builder_config.all_tensors.get();
    auto const& partial_reducer_worker_cores_per_direction = builder_config.worker_cores.get().partial_reducers;
    for (auto direction : {LineDirection::FORWARD, LineDirection::BACKWARD}) {
        TT_FATAL(
            all_program_tensors.remote_output[direction] != nullptr,
            "Internal error. Expected remote output tensor from direction {} to be populated",
            direction);
        TT_FATAL(
            all_program_tensors.input_tensor_from_remote[direction] != nullptr,
            "Internal error. Expected input tensor from remote direction {} to be populated",
            direction);
        TT_ASSERT(
            all_program_tensors.input_tensor->shape() == all_program_tensors.remote_output[direction]->shape(),
            "Input tensor and remote output tensor - direction {} must have the same shape",
            direction);
        TT_ASSERT(
            all_program_tensors.input_tensor->shape() ==
                all_program_tensors.input_tensor_from_remote[direction]->shape(),
            "Input tensor and input from remote tensor from direction {} must have the same shape",
            direction);
    }
    TT_FATAL(
        partial_reducer_worker_cores_per_direction[LineDirection::FORWARD].num_cores() ==
            partial_reducer_worker_cores_per_direction[LineDirection::BACKWARD].num_cores(),
        "Internal error. Expected number of partial reducer workers to be the same for both directions");
}

static void create_non_end_of_line_worker_commands(
    ReduceScatterBuilderConfig& builder_config,
    WorkerCommandStreams& worker_command_streams_out,
    std::unordered_map<CoreCoord, size_t>& math_page_counts_out) {
    validate_non_end_of_line_tensors(builder_config);

    auto const& all_program_tensors = builder_config.all_tensors.get();
    auto const& partial_reducer_worker_cores_per_direction = builder_config.worker_cores.get().partial_reducers;
    auto const& topology_config = builder_config.topology_config.get();

    using namespace ttnn::ccl::worker_detail;
    using namespace ttnn::ccl::cmd;
    using namespace ttnn::ccl::cmd::builder;

    auto const num_workers = partial_reducer_worker_cores_per_direction[LineDirection::FORWARD].num_cores();
    auto const nchips = topology_config.line_size();
    auto const last_chip = topology_config.line_size() - 1;
    // in_tensor_slices = input_tensor.shape.chunk(n=line_size, dim=dim)
    // in_slices_fwd = reverse(in_tensor_slices[topology_config.line_index():]) --> For chip 1, of 4 chip line we want
    // slices 3, 2, 1 in_slices_bwd = in_tensor_slices[:line_toptopology_configology.line_index() + 1] // assuming
    // exclusive end --> For chip 1, of 4 chip line we want slices 0, 1 out_remote_slices_fwd =
    // reverse(in_tensor_slices[topology_config.line_index() + 1:]) --> For chip 1, of 4 chip line we want slices 3, 2
    // out_remote_slices_bwd = in_tensor_slices[topology_config.line_index():]) --> For chip 1, of 4 chip line we want
    // slices 0 (we are only forwarding one slice) Note those that vslice uses inclusive ends so the end values below
    // are off-by-one from the examples above
    auto const input_tensor_slices =
        generate_tensor_slices(nchips, *all_program_tensors.input_tensor, builder_config.dim);
    TT_FATAL(input_tensor_slices.size() == nchips, "Internal error. Expected number of slices to match line size");

    auto const in_slices_fwd = vslice(input_tensor_slices, last_chip, topology_config.line_index());
    auto const in_slices_bwd = vslice(input_tensor_slices, topology_config.line_index(), 0);
    auto const out_remote_slices_fwd = vslice(input_tensor_slices, last_chip, topology_config.line_index() + 1);
    auto const out_remote_slices_bwd = vslice(input_tensor_slices, topology_config.line_index() - 1, 0);

    std::array<std::vector<std::vector<ttnn::ccl::v2::TensorSlice>>, 2> reader_worker_slices_by_direction = {
        split_tensor_slices_across_workers_page_aligned(num_workers, in_slices_fwd),
        split_tensor_slices_across_workers_page_aligned(num_workers, in_slices_bwd)};
    std::array<std::vector<std::vector<ttnn::ccl::v2::TensorSlice>>, 2> writer_worker_slices_by_direction = {
        split_tensor_slices_across_workers_page_aligned(num_workers, out_remote_slices_fwd),
        split_tensor_slices_across_workers_page_aligned(num_workers, out_remote_slices_bwd)};

    // Command stream creation
    populate_partial_reduce_worker_commands(
        builder_config,
        reader_worker_slices_by_direction,
        writer_worker_slices_by_direction,
        worker_command_streams_out,
        math_page_counts_out);

    create_non_end_of_line_final_reducer_worker_commands(
        builder_config, worker_command_streams_out, math_page_counts_out);
}

static void create_worker_runtime_args_not_end_of_line(
    ReduceScatterBuilderConfig& builder_config,
    fabric_lifetime_mode fabric_mode,
    WorkerCommandStreams& worker_command_streams_out,
    std::unordered_map<CoreCoord, size_t>& math_page_counts_out) {
    // Kernel Creation
    create_final_reducer_worker_rt_args_not_end_of_line(
        builder_config, fabric_mode, worker_command_streams_out, math_page_counts_out);

    populate_partial_reduce_rt_args(builder_config, worker_command_streams_out, math_page_counts_out);
}

static void validate_tensors(ProgramTensorsBundle const& all_tensors, LineTopology topology_config) {
    if (topology_config.topology() == ttnn::ccl::Topology::Linear) {
        const size_t page_size = get_page_size(*all_tensors.input_tensor);
        for (auto direction : {LineDirection::FORWARD, LineDirection::BACKWARD}) {
            if (!topology_config.is_at_end_of_line()) {
                TT_FATAL(all_tensors.remote_output[direction] != nullptr, "Remote output tensor must be populated");
                TT_FATAL(
                    page_size == get_page_size(*all_tensors.remote_output[direction]),
                    "Remote output tensor must have the same page size as input tensor");
            }
            if (topology_config.is_first_device_in_line(direction)) {
                TT_FATAL(
                    all_tensors.local_output_partial[direction] != nullptr,
                    "Local output partial tensor must be populated");
                TT_FATAL(
                    all_tensors.input_tensor_from_remote[direction] == nullptr,
                    "Input tensor from remote must be populated");
                TT_FATAL(all_tensors.remote_output[direction] != nullptr, "Remote output tensor must be populated");
                TT_FATAL(
                    page_size == get_page_size(*all_tensors.remote_output[direction]),
                    "Remote output tensor must have the same page size as input tensor");
            } else if (topology_config.is_last_device_in_line(direction)) {
                TT_FATAL(
                    all_tensors.input_tensor_from_remote[direction] != nullptr,
                    "Input tensor from remote must be populated");
                TT_FATAL(all_tensors.remote_output[direction] == nullptr, "Remote output tensor must be populated");
                TT_FATAL(
                    page_size == get_page_size(*all_tensors.input_tensor_from_remote[direction]),
                    "Input tensor from remote must have the same page size as input tensor");
            }
            if (all_tensors.local_output_partial[direction] != nullptr) {
                TT_FATAL(
                    all_tensors.local_output_partial[direction]->shape() == all_tensors.local_output_tensor->shape(),
                    "Partial output tensor and local output tensor must have the same shape");
            }
            if (all_tensors.input_tensor_from_remote[direction] != nullptr) {
                TT_FATAL(
                    all_tensors.input_tensor_from_remote[direction]->shape() == all_tensors.input_tensor->shape(),
                    "Input tensor from remote and input tensor must have the same shape");
            }
            if (all_tensors.remote_output[direction] != nullptr) {
                TT_FATAL(
                    all_tensors.remote_output[direction]->shape() == all_tensors.input_tensor->shape(),
                    "Remote output tensor and input tensor must have the same shape");
            }
        }
    } else {
        return;
    }
}

static void initialize_op_internal_tensor_syncs(
    Program& program,
    Device* device,
    std::array<Device*, 2> const& neighbour_devices,
    ProgramTensorsBundle& all_tensors,
    WorkerCoreBundle const& worker_cores,
    std::shared_ptr<const GlobalSemaphore> const& from_remote_sem,
    std::shared_ptr<const GlobalSemaphore> const& to_remote_sem) {
    auto core_coord_lt = [](CoreCoord const& a, CoreCoord const& b) { return a.y < b.y || (a.y == b.y && a.x < b.x); };

    TT_FATAL(
        worker_cores.partial_reducers_vec[LineDirection::BACKWARD].size() > 0,
        "Internal error. Expected at least one partial reducer worker");
    std::array<std::vector<CoreCoord>, 2> partial_reducer_cores = {
        worker_cores.partial_reducers_vec[LineDirection::FORWARD],
        worker_cores.partial_reducers_vec[LineDirection::BACKWARD]};
    auto all_partial_reducer_cores = worker_cores.partial_reducers[LineDirection::FORWARD];
    all_partial_reducer_cores = all_partial_reducer_cores.merge(worker_cores.partial_reducers[LineDirection::BACKWARD]);

    auto partial_reducers_in1_sem_id = CreateSemaphore(program, all_partial_reducer_cores, 0, CoreType::WORKER);
    for (auto direction : {LineDirection::FORWARD, LineDirection::BACKWARD}) {
        all_tensors.input_tensor_from_remote_sync[direction] = TensorSyncSpec{};
        for (auto const& worker_core : partial_reducer_cores[direction]) {
            all_tensors.input_tensor_from_remote_sync[direction].targets.push_back(TensorSyncSpec::target_rect{
                device->worker_core_from_logical_core(worker_core).x,
                device->worker_core_from_logical_core(worker_core).y,
                device->worker_core_from_logical_core(worker_core).x,
                device->worker_core_from_logical_core(worker_core).y,
            });
            all_tensors.input_tensor_from_remote_sync[direction].semaphore_ids.push_back(from_remote_sem.get());
            all_tensors.input_tensor_from_remote_sync[direction].completion_target_value_per_semaphore.push_back(1);

            // remote output sync
            if (neighbour_devices[direction] != nullptr) {
                all_tensors.remote_output_sync[direction].semaphore_ids.push_back(to_remote_sem.get());// = all_tensors.input_tensor_from_remote_sync[direction];
                all_tensors.remote_output_sync[direction].completion_target_value_per_semaphore.push_back(1);// = all_tensors.input_tensor_from_remote_sync[direction];
                all_tensors.remote_output_sync[direction] = all_tensors.input_tensor_from_remote_sync[direction];
                all_tensors.remote_output_sync[direction].targets.back() = TensorSyncSpec::target_rect{
                    neighbour_devices[direction]->worker_core_from_logical_core(worker_core).x,
                    neighbour_devices[direction]->worker_core_from_logical_core(worker_core).y,
                    neighbour_devices[direction]->worker_core_from_logical_core(worker_core).x,
                    neighbour_devices[direction]->worker_core_from_logical_core(worker_core).y,
                };
            }
        }
    }

    auto final_reducer_cores = corerange_to_cores(worker_cores.final_reducers, std::nullopt, true);
    std::array<uint32_t, 2> final_reducer_partial_input_sem_ids = {
        CreateSemaphore(program, worker_cores.final_reducers, 0, CoreType::WORKER),
        CreateSemaphore(program, worker_cores.final_reducers, 0, CoreType::WORKER)};
    for (auto const& worker_core : final_reducer_cores) {
        auto worker_target = TensorSyncSpec::target_rect{
            device->worker_core_from_logical_core(worker_core).x,
            device->worker_core_from_logical_core(worker_core).y,
            device->worker_core_from_logical_core(worker_core).x,
            device->worker_core_from_logical_core(worker_core).y,
        };
        all_tensors.local_output_partial_sync[LineDirection::FORWARD].targets.push_back(worker_target);
        all_tensors.local_output_partial_sync[LineDirection::FORWARD].completion_target_value_per_semaphore.push_back(
            1);
        all_tensors.local_output_partial_sync[LineDirection::FORWARD].semaphore_ids.push_back(
            final_reducer_partial_input_sem_ids[LineDirection::FORWARD]);
        all_tensors.local_output_partial_sync[LineDirection::BACKWARD].targets.push_back(worker_target);
        all_tensors.local_output_partial_sync[LineDirection::BACKWARD].completion_target_value_per_semaphore.push_back(
            1);
        all_tensors.local_output_partial_sync[LineDirection::BACKWARD].semaphore_ids.push_back(
            final_reducer_partial_input_sem_ids[LineDirection::BACKWARD]);
    }

    for (auto direction : {LineDirection::FORWARD, LineDirection::BACKWARD}) {
        TT_FATAL(
            all_tensors.input_tensor_from_remote_sync[direction].targets.size() > 0,
            "Input tensor from remote sync must be populated");
        TT_FATAL(
            all_tensors.input_tensor_from_remote_sync[direction].semaphore_ids.size() > 0,
            "Input tensor from remote sync must be populated");
        TT_FATAL(
            all_tensors.input_tensor_from_remote_sync[direction].completion_target_value_per_semaphore.size() > 0,
            "Input tensor from remote sync must be populated");
        TT_FATAL(
            all_tensors.input_tensor_from_remote_sync[direction].completion_target_value_per_semaphore.size() ==
                all_tensors.input_tensor_from_remote_sync[direction].semaphore_ids.size(),
            "Input tensor from remote sync must be populated");

        TT_FATAL(
            all_tensors.remote_output_sync[direction].completion_target_value_per_semaphore.size() ==
                all_tensors.remote_output_sync[direction].semaphore_ids.size(),
            "Remote output sync must be populated");

        TT_FATAL(
            all_tensors.local_output_partial_sync[direction].targets.size() > 0,
            "Local output partial sync must be populated");
        TT_FATAL(
            all_tensors.local_output_partial_sync[direction].semaphore_ids.size() > 0,
            "Local output partial sync must be populated");
        TT_FATAL(
            all_tensors.local_output_partial_sync[direction].completion_target_value_per_semaphore.size() > 0,
            "Local output partial sync must be populated");
        TT_FATAL(
            all_tensors.local_output_partial_sync[direction].completion_target_value_per_semaphore.size() ==
                all_tensors.local_output_partial_sync[direction].semaphore_ids.size(),
            "Local output partial sync must be populated");
    }
    TT_FATAL(
        all_tensors.remote_output_sync[LineDirection::FORWARD].targets.size() > 0 ||
            all_tensors.remote_output_sync[LineDirection::BACKWARD].targets.size() > 0,
        "Remote output sync must be populated");
    TT_FATAL(
        all_tensors.remote_output_sync[LineDirection::FORWARD].semaphore_ids.size() > 0 ||
            all_tensors.remote_output_sync[LineDirection::BACKWARD].semaphore_ids.size() > 0,
        "Remote output sync must be populated");
    TT_FATAL(
        all_tensors.remote_output_sync[LineDirection::FORWARD].completion_target_value_per_semaphore.size() > 0 ||
            all_tensors.remote_output_sync[LineDirection::BACKWARD].completion_target_value_per_semaphore.size() > 0,
        "Remote output sync must be populated");
}

static void generate_worker_command_streams(
    ReduceScatterBuilderConfig& builder_config,
    fabric_lifetime_mode fabric_mode,
    WorkerCommandStreams& command_streams,
    std::unordered_map<CoreCoord, size_t>& math_page_counts) {
    bool is_end_of_line = builder_config.topology_config.get().is_at_end_of_line();
    if (is_end_of_line) {
        create_end_of_line_worker_commands(builder_config, fabric_mode, command_streams, math_page_counts);
    } else {
        create_non_end_of_line_worker_commands(builder_config, command_streams, math_page_counts);
    }
}



static void populate_worker_runtime_args(
    ReduceScatterBuilderConfig& builder_config,
    fabric_lifetime_mode fabric_mode,
    WorkerCommandStreams& command_streams,
    std::unordered_map<CoreCoord, size_t>& math_page_counts) {
    bool is_end_of_line = builder_config.topology_config.get().is_at_end_of_line();
    if (is_end_of_line) {
        create_worker_runtime_args_for_inactive_workers(builder_config);
        create_end_of_line_worker_runtime_args(builder_config, command_streams, math_page_counts);
    } else {
        create_worker_runtime_args_not_end_of_line(builder_config, fabric_mode, command_streams, math_page_counts);
    }
}

operation::ProgramWithCallbacks reduce_scatter_async_on_instantiated_edm_fabric(
    Program& program,
    ttnn::ccl::EdmLineFabricOpInterface& fabric,
    std::optional<Device*> forward_device,
    std::optional<Device*> backward_device,
    Tensor const& input_tensor,
    Tensor& local_output_tensor,
    Tensor& input_tensor_from_remote_forward_direction,
    Tensor& input_tensor_from_remote_backward_direction,
    Tensor& partial_output_tensor_to_forward_direction,
    Tensor& partial_output_tensor_to_backward_direction,
    std::optional<Tensor>& foreward_direction_remote_output_tensor,
    std::optional<Tensor>& backward_direction_remote_output_tensor,
    ttnn::operations::binary::BinaryOpType reduce_op,
    size_t line_size,
    size_t line_index,
    const uint32_t dim,
    const size_t num_links,
    ttnn::ccl::Topology topology,

    fabric_lifetime_mode fabric_mode,
    std::shared_ptr<const GlobalSemaphore> const& from_remote_sems,
    std::shared_ptr<const GlobalSemaphore> const& to_remote_sem) {
    using namespace ttnn::ccl::worker_detail;
    bool do_dynamic_fabric_bringup_and_teardown = fabric_mode == fabric_lifetime_mode::TRANSIENT;

    // Constants/ "Globals"
    constexpr auto math_in0_cb = tt::CBIndex::c_0;
    constexpr auto math_in1_cb = tt::CBIndex::c_1;
    constexpr auto math_out_cb = tt::CBIndex::c_2;
    constexpr auto pass_through_cb = tt::CBIndex::c_3;
    AllReduceScatterCircularBufferIds all_cbs = {
        {pass_through_cb, math_in0_cb, math_in1_cb},
        {pass_through_cb, math_out_cb},
        {math_in0_cb, math_in1_cb},
        {math_out_cb},
        {pass_through_cb},
        {pass_through_cb},
        {math_in0_cb, math_in1_cb},
        {math_out_cb}};

    const size_t page_size = get_page_size(input_tensor);
    Device* device = input_tensor.device();
    std::array<Device*, 2> neighbour_devices = {forward_device.value_or(nullptr), backward_device.value_or(nullptr)};
    size_t fabric_buffer_size_pages = fabric.get_edm_buffer_size_bytes() / get_page_size(input_tensor);
    auto const& topology_config = LineTopology(line_size, line_index);

    auto const& worker_cores = select_worker_cores(topology, num_links);//, device);
    ProgramTensorsBundle all_tensors = {
        ProgramTensorsBundle::build_handle(input_tensor),
        {},

        ProgramTensorsBundle::build_handle(local_output_tensor),
        {},

        {topology_config.is_first_device_in_line(LineDirection::FORWARD)
             ? nullptr
             : ProgramTensorsBundle::build_handle(input_tensor_from_remote_forward_direction),
         topology_config.is_first_device_in_line(LineDirection::BACKWARD)
             ? nullptr
             : ProgramTensorsBundle::build_handle(input_tensor_from_remote_backward_direction)},
        {},

        {topology_config.is_last_device_in_line(LineDirection::FORWARD)
             ? nullptr
             : ProgramTensorsBundle::build_handle(input_tensor_from_remote_backward_direction),
         topology_config.is_last_device_in_line(LineDirection::BACKWARD)
             ? nullptr
             : ProgramTensorsBundle::build_handle(input_tensor_from_remote_forward_direction)},
        {},

        {ProgramTensorsBundle::build_handle(partial_output_tensor_to_forward_direction),
         ProgramTensorsBundle::build_handle(partial_output_tensor_to_backward_direction)},
        {}};


    initialize_op_internal_tensor_syncs(
        program, device, neighbour_devices, all_tensors, worker_cores, from_remote_sems, to_remote_sem);

    validate_tensors(all_tensors, topology_config);

    // Circular Buffer Creation
    size_t const cb_page_size = page_size + sizeof(tt::fabric::PacketHeader);
    auto const cb_handles = create_worker_circular_buffers(
        program,
        worker_cores.all_worker_cores,
        math_in0_cb,
        math_in1_cb,
        math_out_cb,
        pass_through_cb,
        fabric_buffer_size_pages,
        // TODO: Move packet headers to side buffer and don't force it through
        page_size + sizeof(tt::fabric::PacketHeader));

    auto kernel_ids =
        build_line_reduce_scatter_worker_ct(program, all_tensors, cb_handles, worker_cores.all_worker_cores, reduce_op);

    const size_t pages_per_cb_packet = fabric.get_edm_buffer_size_bytes() / cb_page_size;
    auto builder_config = ReduceScatterBuilderConfig{
        program,
        device,
        forward_device.value_or(nullptr),
        backward_device.value_or(nullptr),
        fabric,
        all_tensors,
        kernel_ids,
        all_cbs,
        topology_config,
        worker_cores,
        page_size,
        pages_per_cb_packet,
        dim};
    bool is_end_of_line = topology_config.is_at_end_of_line();

    log_trace(tt::LogOp, "Pages per CB packet: {}", pages_per_cb_packet);
    WorkerCommandStreams command_streams;
    std::unordered_map<CoreCoord, size_t> math_page_counts;
    generate_worker_command_streams(builder_config, fabric_mode, command_streams, math_page_counts);

    populate_worker_runtime_args(builder_config, fabric_mode, command_streams, math_page_counts);


    // Synchronous mode kernel invocation
    auto override_runtime_arguments_callback =
        [topology_config, from_remote_sems, to_remote_sem, kernel_ids, worker_cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors.at(0);
            const auto& output = output_tensors.at(0);
            auto& worker_reader_runtime_args_by_core = GetRuntimeArgs(program, kernel_ids.reader);
            auto& worker_writer_runtime_args_by_core = GetRuntimeArgs(program, kernel_ids.writer);
        };

    log_trace(tt::LogOp, "Done program factory");

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks build_reduce_scatter_async_program(
    Tensor const& input_tensor,
    Tensor& local_output_tensor,
    Tensor& input_tensor_from_remote_forward_direction,
    Tensor& input_tensor_from_remote_backward_direction,
    Tensor& partial_output_tensor_to_forward_direction,
    Tensor& partial_output_tensor_to_backward_direction,
    std::optional<Tensor>& foreward_direction_remote_output_tensor,
    std::optional<Tensor>& backward_direction_remote_output_tensor,
    std::optional<Device*> forward_device,
    std::optional<Device*> backward_device,
    ttnn::operations::binary::BinaryOpType reduce_op,
    const uint32_t dim,
    const uint32_t line_size,
    const uint32_t line_index,
    ttnn::ccl::Topology topology,
    std::optional<size_t> num_links_preferred,
    std::shared_ptr<const tt::tt_metal::GlobalSemaphore> const& from_remote_sem,
    std::shared_ptr<const tt::tt_metal::GlobalSemaphore> const& to_remote_sem,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle_) {
    auto program = tt::tt_metal::Program();

    bool persistent_fabric = true;
    Device* device = input_tensor.device();

    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle = fabric_handle_;
    fabric_lifetime_mode fabric_mode = fabric_lifetime_mode::PERSISTENT;
        // fabric_handle.has_value() ? fabric_lifetime_mode::PERSISTENT : fabric_lifetime_mode::TRANSIENT;
    // We only build the local chip's part of the fabric
    if (!fabric_handle.has_value()) {
        fabric_handle = ttnn::ccl::EdmLineFabricOpInterface(
            device,
            forward_device,
            backward_device,
            &program,
            persistent_fabric,
            num_links_preferred.value_or(line_size));
    }

    TT_FATAL(fabric_mode == fabric_lifetime_mode::PERSISTENT, "Reduce scatter doesn't support transient fabric mode");
    return reduce_scatter_async_on_instantiated_edm_fabric(
        program,
        fabric_handle.value(),
        forward_device,
        backward_device,
        input_tensor,
        local_output_tensor,
        input_tensor_from_remote_forward_direction,
        input_tensor_from_remote_backward_direction,
        partial_output_tensor_to_forward_direction,
        partial_output_tensor_to_backward_direction,
        foreward_direction_remote_output_tensor,
        backward_direction_remote_output_tensor,
        reduce_op,
        line_size,
        line_index,
        dim,
        fabric_handle.value().get_num_links(),
        ttnn::ccl::Topology::Linear,
        fabric_mode,
        from_remote_sem,
        to_remote_sem);
}

}  // namespace ttnn::ccl::reduce_scatter_detail
