// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <device_command.hpp>
#include <device.hpp>
#include "assert.hpp"
#include "dispatch.hpp"
#include <tt-metalium/command_queue_interface.hpp>
#include <tt-metalium/dispatch_settings.hpp>

namespace tt::tt_metal {
namespace buffer_dispatch {

// ====== Utility Functions for Writes ======

// Dispatch constants required for writing buffer data
struct BufferDispatchConstants {
    uint32_t issue_queue_cmd_limit = 0;
    uint32_t max_prefetch_cmd_size = 0;
    uint32_t max_data_sizeB = 0;
};

// Dispatch parameters computed during runtime. These are used
// to assemble dispatch commands and compute src + dst offsets
// required to write buffer data.
struct BufferWriteDispatchParams {
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    uint32_t address = 0;
    uint32_t dst_page_index = 0;
    uint32_t page_size_to_write = 0;
    uint32_t total_pages_to_write = 0;
    uint32_t total_pages_written = 0;
    uint32_t pages_per_txn = 0;
    bool issue_wait = false;
    IDevice* device = nullptr;
    uint32_t cq_id = 0;
};

// Parameters specific to interleaved buffers
struct InterleavedBufferWriteDispatchParams : BufferWriteDispatchParams {
    uint32_t num_banks = 0;
    const Buffer& buffer;

    InterleavedBufferWriteDispatchParams(
        const Buffer& buffer,
        uint32_t dst_page_index,
        uint32_t total_pages_to_write,
        uint32_t cq_id,
        tt::stl::Span<const uint32_t> expected_num_workers_completed) :
        buffer(buffer) {
        this->num_banks = buffer.device()->allocator()->get_num_banks(buffer.buffer_type());
        this->address = buffer.address();
        this->dst_page_index = dst_page_index;
        this->page_size_to_write = buffer.aligned_page_size();
        this->total_pages_to_write = total_pages_to_write;
        this->device = buffer.device();
        this->cq_id = cq_id;
        this->expected_num_workers_completed = expected_num_workers_completed;
    }
    virtual ~InterleavedBufferWriteDispatchParams() = default;

    void calculate_issue_wait() {
        this->issue_wait = this->total_pages_written == 0;  // only stall for the first write of the buffer
    }

    virtual void calculate_num_pages_for_write_transaction(uint32_t num_pages_available_in_cq) {
        this->pages_per_txn = std::min(this->total_pages_to_write, num_pages_available_in_cq);
    }

    virtual bool is_page_offset_out_of_bounds() const {
        return this->dst_page_index > CQ_DISPATCH_CMD_PAGED_WRITE_MAX_PAGE_INDEX;
    }

    // Page offset in CQ_DISPATCH_CMD_WRITE_PAGED is uint16_t
    // To handle larger page offsets move bank base address up and update page offset to be relative to the new
    // bank address
    virtual void update_params_to_be_within_bounds() {
        const uint32_t num_pages_written_per_bank = this->dst_page_index / this->num_banks;
        this->address += num_pages_written_per_bank * this->page_size_to_write;
        this->dst_page_index %= this->num_banks;
    }

    virtual void update_params_after_write_transaction() {
        this->total_pages_to_write -= this->pages_per_txn;
        this->total_pages_written += this->pages_per_txn;
        this->dst_page_index += this->pages_per_txn;
    }

    virtual bool write_large_pages() const { return false; }

    virtual uint32_t num_full_pages_written() const { return this->total_pages_written; }

    virtual uint32_t num_partial_pages_per_full_page() const { return 1; }
};

struct InterleavedBufferWriteLargePageDispatchParams : InterleavedBufferWriteDispatchParams {
    InterleavedBufferWriteLargePageDispatchParams(
        const Buffer& buffer,
        uint32_t dst_page_index,
        uint32_t page_size_to_write,
        uint32_t total_pages_to_write,
        uint32_t num_full_pages,
        uint32_t cq_id,
        tt::stl::Span<const uint32_t> expected_num_workers_completed) :
        InterleavedBufferWriteDispatchParams(
            buffer, dst_page_index, total_pages_to_write, cq_id, expected_num_workers_completed) {
        this->page_size_to_write = page_size_to_write;
        this->full_pages_to_write = num_full_pages;
        this->full_page_size = buffer.aligned_page_size();
        this->num_partial_pages_in_single_full_page = full_page_size / page_size_to_write;
    }

    void calculate_num_pages_for_write_transaction(uint32_t num_pages_available_in_cq) override {
        TT_ASSERT(this->num_banks > this->dst_page_index);
        this->pages_per_txn =
            std::min({this->full_pages_to_write, this->num_banks - this->dst_page_index, num_pages_available_in_cq});
    }

    bool is_page_offset_out_of_bounds() const override { return this->dst_page_index >= this->num_banks; }

    void update_params_to_be_within_bounds() override {
        const uint32_t num_pages_written_per_bank = this->dst_page_index / this->num_banks;
        this->address += num_pages_written_per_bank * this->full_page_size;
        this->dst_page_index %= this->num_banks;
    }

    void update_params_after_write_transaction() override {
        this->total_pages_to_write -= this->pages_per_txn;
        this->total_pages_written += this->pages_per_txn;
        this->address += this->page_size_to_write;
        if (this->were_full_pages_written_in_last_write_transaction()) {
            this->full_pages_to_write -= this->pages_per_txn;
            this->full_pages_written += this->pages_per_txn;
            if (!this->will_next_full_page_be_round_robined()) {
                this->address -= this->full_page_size;
            }
            this->dst_page_index += this->pages_per_txn;
            this->dst_page_index %= this->num_banks;
        }
    }

    bool write_large_pages() const override { return true; }

    uint32_t num_full_pages_written() const override { return this->full_pages_written; }

    uint32_t num_partial_pages_per_full_page() const override { return this->num_partial_pages_in_single_full_page; }

private:
    uint32_t num_partial_pages_in_single_full_page = 0;
    uint32_t full_page_size = 0;
    uint32_t full_pages_written = 0;
    uint32_t full_pages_to_write = 0;

    bool were_full_pages_written_in_last_write_transaction() const {
        const uint32_t page_size = this->address - this->buffer.address();
        return page_size > 0 && page_size % this->full_page_size == 0;
    }

    bool will_next_full_page_be_round_robined() const {
        const uint32_t dst_page_index_next_txn = this->dst_page_index + this->pages_per_txn;
        return dst_page_index_next_txn != (dst_page_index_next_txn % this->num_banks);
    }
};

// Parameters specific to sharded buffers
struct ShardedBufferWriteDispatchParams : BufferWriteDispatchParams {
    bool width_split = false;
    uint32_t starting_dst_host_page_index = 0;
    uint32_t initial_pages_skipped = 0;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping = nullptr;
    uint32_t max_pages_per_shard = 0;
    CoreCoord core;
};

void update_byte_offset_in_cq(uint32_t& byte_offset, bool issue_wait) {
    if (issue_wait) {
        byte_offset *= 2;  // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
    }
}

int32_t calculate_num_pages_available_in_cq(
    const InterleavedBufferWriteDispatchParams& dispatch_params,
    const BufferDispatchConstants& dispatch_constants,
    uint32_t byte_offset_in_cq) {
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t space_availableB = std::min(
        dispatch_constants.issue_queue_cmd_limit - sysmem_manager.get_issue_queue_write_ptr(dispatch_params.cq_id),
        dispatch_constants.max_prefetch_cmd_size);
    int32_t num_pages_available =
        (int32_t(space_availableB) - int32_t(byte_offset_in_cq)) / int32_t(dispatch_params.page_size_to_write);
    return num_pages_available;
}

// Generate dispatch constants
BufferDispatchConstants generate_buffer_dispatch_constants(
    const SystemMemoryManager& sysmem_manager, CoreType dispatch_core_type, uint32_t cq_id) {
    BufferDispatchConstants buf_dispatch_constants;

    buf_dispatch_constants.issue_queue_cmd_limit = sysmem_manager.get_issue_queue_limit(cq_id);
    buf_dispatch_constants.max_prefetch_cmd_size = DispatchMemMap::get(dispatch_core_type).max_prefetch_command_size();
    buf_dispatch_constants.max_data_sizeB = buf_dispatch_constants.max_prefetch_cmd_size -
                                            (hal.get_alignment(HalMemType::HOST) * 2);  // * 2 to account for issue

    return buf_dispatch_constants;
}

// Initialize Dispatch Parameters - reused across write txns
ShardedBufferWriteDispatchParams initialize_sharded_buf_dispatch_params(
    Buffer& buffer,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferDispatchConstants& buf_dispatch_constants,
    const BufferRegion& region) {
    ShardedBufferWriteDispatchParams dispatch_params;
    dispatch_params.width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
    dispatch_params.buffer_page_mapping = (dispatch_params.width_split) ? buffer.get_buffer_page_mapping() : nullptr;
    dispatch_params.total_pages_to_write = region.size / buffer.page_size();
    dispatch_params.total_pages_written = 0;
    dispatch_params.max_pages_per_shard = buffer.shard_spec().size();
    dispatch_params.page_size_to_write = buffer.aligned_page_size();
    dispatch_params.dst_page_index = region.offset / buffer.page_size();
    dispatch_params.starting_dst_host_page_index = region.offset / buffer.page_size();
    dispatch_params.initial_pages_skipped = 0;
    dispatch_params.device = buffer.device();
    dispatch_params.cq_id = cq_id;
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;

    TT_FATAL(
        buf_dispatch_constants.max_data_sizeB >= dispatch_params.page_size_to_write,
        "Writing padded page size > {} is currently unsupported for sharded tensors.",
        buf_dispatch_constants.max_data_sizeB);
    return dispatch_params;
}

std::unique_ptr<InterleavedBufferWriteDispatchParams> initialize_interleaved_buf_dispatch_params(
    Buffer& buffer,
    const BufferDispatchConstants& buf_dispatch_constants,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferRegion& region) {
    std::unique_ptr<InterleavedBufferWriteDispatchParams> dispatch_params;

    uint32_t total_pages_to_write = region.size / buffer.page_size();
    const uint32_t dst_page_index = region.offset / buffer.page_size();

    const bool write_large_pages = buffer.aligned_page_size() > buf_dispatch_constants.max_data_sizeB;
    if (write_large_pages) {
        uint32_t partial_page_size = DispatchSettings::BASE_PARTIAL_PAGE_SIZE;
        const uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
        while (buffer.aligned_page_size() % partial_page_size != 0) {
            partial_page_size += pcie_alignment;
        }
        const uint32_t page_size_to_write = partial_page_size;
        const uint32_t padded_buffer_size = total_pages_to_write * buffer.aligned_page_size();
        const uint32_t num_full_pages = total_pages_to_write;
        total_pages_to_write = padded_buffer_size / page_size_to_write;
        dispatch_params = std::make_unique<InterleavedBufferWriteLargePageDispatchParams>(
            buffer,
            dst_page_index,
            page_size_to_write,
            total_pages_to_write,
            num_full_pages,
            cq_id,
            expected_num_workers_completed);
    } else {
        dispatch_params = std::make_unique<InterleavedBufferWriteDispatchParams>(
            buffer, dst_page_index, total_pages_to_write, cq_id, expected_num_workers_completed);
    }

    return dispatch_params;
}

// Populate/Assemble dispatch commands for writing buffer data
void populate_interleaved_buffer_write_dispatch_cmds(
    const void* src,
    HugepageDeviceCommand& command_sequence,
    Buffer& buffer,
    InterleavedBufferWriteDispatchParams& dispatch_params) {
    const uint8_t is_dram = uint8_t(buffer.is_dram());
    TT_ASSERT(
        dispatch_params.dst_page_index <= CQ_DISPATCH_CMD_PAGED_WRITE_MAX_PAGE_INDEX,
        "Page offset needs to fit within range of uint16_t, bank_base_address was computed incorrectly!");
    const uint16_t start_page = uint16_t(dispatch_params.dst_page_index & CQ_DISPATCH_CMD_PAGED_WRITE_MAX_PAGE_INDEX);
    const bool flush_prefetch = true;
    command_sequence.add_dispatch_write_paged(
        flush_prefetch,
        is_dram,
        start_page,
        dispatch_params.address,
        dispatch_params.page_size_to_write,
        dispatch_params.pages_per_txn);

    const uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;

    // TODO: Consolidate
    if (dispatch_params.write_large_pages()) {
        const uint32_t num_full_pages_written = dispatch_params.num_full_pages_written();
        const uint32_t num_partial_pages_written = dispatch_params.total_pages_written;
        const uint32_t num_partial_pages_per_full_page = dispatch_params.num_partial_pages_per_full_page();
        const uint32_t num_partial_pages_written_associated_with_current_full_pages =
            num_partial_pages_written - (num_full_pages_written * num_partial_pages_per_full_page);
        const uint32_t num_partial_pages_written_per_current_full_page =
            num_partial_pages_written_associated_with_current_full_pages / dispatch_params.pages_per_txn;
        uint32_t num_partial_pages_written_curr_txn = 0;
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
             sysmem_address_offset += dispatch_params.page_size_to_write) {
            uint32_t page_size_to_copy = dispatch_params.page_size_to_write;
            uint32_t src_address_offset = num_full_pages_written * buffer.page_size() +
                                          num_partial_pages_written_per_current_full_page * page_size_to_copy +
                                          num_partial_pages_written_curr_txn * buffer.page_size();
            if (num_partial_pages_written_per_current_full_page == num_partial_pages_per_full_page - 1) {
                // last partial page being copied from unpadded src buffer
                const uint32_t padding = buffer.aligned_page_size() - buffer.page_size();
                page_size_to_copy -= padding;
            }
            command_sequence.add_data(
                (char*)src + src_address_offset, page_size_to_copy, dispatch_params.page_size_to_write);
            num_partial_pages_written_curr_txn += 1;
        }
    } else {
        uint32_t src_address_offset = dispatch_params.total_pages_written * buffer.page_size();
        if (buffer.page_size() % buffer.alignment() != 0 and buffer.page_size() != buffer.size()) {
            // If page size is not aligned, we cannot do a contiguous write
            for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
                 sysmem_address_offset += dispatch_params.page_size_to_write) {
                command_sequence.add_data(
                    (char*)src + src_address_offset, buffer.page_size(), dispatch_params.page_size_to_write);
                src_address_offset += buffer.page_size();
            }
        } else {
            command_sequence.add_data((char*)src + src_address_offset, data_size_bytes, data_size_bytes);
        }
    }
}

void populate_sharded_buffer_write_dispatch_cmds(
    const void* src,
    HugepageDeviceCommand& command_sequence,
    Buffer& buffer,
    ShardedBufferWriteDispatchParams& dispatch_params) {
    uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;
    auto noc_index = dispatch_downstream_noc;
    const CoreCoord virtual_core =
        buffer.device()->virtual_core_from_logical_core(dispatch_params.core, buffer.core_type());
    command_sequence.add_dispatch_write_linear(
        0,
        buffer.device()->get_noc_unicast_encoding(noc_index, virtual_core),
        dispatch_params.address,
        data_size_bytes);

    if (dispatch_params.width_split) {
        TT_ASSERT(dispatch_params.buffer_page_mapping != nullptr);
        const auto& page_mapping = *(dispatch_params.buffer_page_mapping);
        uint8_t* dst = command_sequence.reserve_space<uint8_t*, true>(data_size_bytes);
        // TODO: Expose getter for cmd_write_offsetB?
        uint32_t dst_offset = dst - (uint8_t*)command_sequence.data();
        for (uint32_t dev_page = dispatch_params.dst_page_index;
             dev_page < dispatch_params.dst_page_index + dispatch_params.pages_per_txn;
             ++dev_page) {
            auto& host_page = page_mapping.dev_page_to_host_page_mapping_[dev_page];
            if (host_page.has_value()) {
                const uint32_t src_offset =
                    (host_page.value() - dispatch_params.starting_dst_host_page_index) * buffer.page_size();
                command_sequence.update_cmd_sequence(dst_offset, (char*)(src) + src_offset, buffer.page_size());
            }
            dst_offset += dispatch_params.page_size_to_write;
        }
    } else {
        uint32_t unpadded_src_offset = dispatch_params.total_pages_written * buffer.page_size();
        if (buffer.page_size() != dispatch_params.page_size_to_write and buffer.page_size() != buffer.size()) {
            for (uint32_t i = 0; i < dispatch_params.pages_per_txn; ++i) {
                command_sequence.add_data(
                    (char*)src + unpadded_src_offset, buffer.page_size(), dispatch_params.page_size_to_write);
                unpadded_src_offset += buffer.page_size();
            }
        } else {
            command_sequence.add_data((char*)src + unpadded_src_offset, data_size_bytes, data_size_bytes);
        }
    }
}

// Issue dispatch commands for writing buffer data
template <typename T>
void issue_buffer_dispatch_command_sequence(
    const void* src,
    Buffer& buffer,
    T& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type) {
    uint32_t num_worker_counters = sub_device_ids.size();
    uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmd_sequence_sizeB = align(
        sizeof(CQPrefetchCmd) +      // CQ_PREFETCH_CMD_RELAY_INLINE
            sizeof(CQDispatchCmd) +  // CQ_DISPATCH_CMD_WRITE_PAGED or CQ_DISPATCH_CMD_WRITE_LINEAR
            data_size_bytes,
        pcie_alignment);
    if (dispatch_params.issue_wait) {
        cmd_sequence_sizeB += hal.get_alignment(HalMemType::HOST) *
                              num_worker_counters;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
    }
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    if (dispatch_params.issue_wait) {
        uint32_t dispatch_message_base_addr =
            DispatchMemMap::get(dispatch_core_type)
                .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
        for (const auto& sub_device_id : sub_device_ids) {
            auto offset_index = sub_device_id.to_index();
            uint32_t dispatch_message_addr =
                dispatch_message_base_addr +
                DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
            command_sequence.add_dispatch_wait(
                false, dispatch_message_addr, dispatch_params.expected_num_workers_completed[offset_index]);
        }
    }
    if constexpr (std::is_same_v<T, ShardedBufferWriteDispatchParams>) {
        populate_sharded_buffer_write_dispatch_cmds(src, command_sequence, buffer, dispatch_params);
    } else {
        populate_interleaved_buffer_write_dispatch_cmds(src, command_sequence, buffer, dispatch_params);
    }

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
}

// Top level helper functions to write buffer data
void write_interleaved_buffer_to_device(
    const void* src,
    InterleavedBufferWriteDispatchParams& dispatch_params,
    Buffer& buffer,
    const BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type) {
    uint32_t byte_offset_in_cq =
        hal.get_alignment(HalMemType::HOST);  // data appended after CQ_PREFETCH_CMD_RELAY_INLINE
                                              // + CQ_DISPATCH_CMD_WRITE_PAGED
    while (dispatch_params.total_pages_to_write > 0) {
        dispatch_params.calculate_issue_wait();
        update_byte_offset_in_cq(byte_offset_in_cq, dispatch_params.issue_wait);

        if (dispatch_params.is_page_offset_out_of_bounds()) {
            dispatch_params.update_params_to_be_within_bounds();
        }

        const int32_t num_pages_available_in_cq =
            calculate_num_pages_available_in_cq(dispatch_params, buf_dispatch_constants, byte_offset_in_cq);
        if (num_pages_available_in_cq <= 0) {
            SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
            sysmem_manager.wrap_issue_queue_wr_ptr(dispatch_params.cq_id);
            continue;
        }

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for command queue {}", dispatch_params.cq_id);

        dispatch_params.calculate_num_pages_for_write_transaction(num_pages_available_in_cq);
        issue_buffer_dispatch_command_sequence(src, buffer, dispatch_params, sub_device_ids, dispatch_core_type);
        dispatch_params.update_params_after_write_transaction();
    }
}

std::vector<CoreCoord> get_cores_for_sharded_buffer(
    bool width_split, const std::shared_ptr<const BufferPageMapping>& buffer_page_mapping, Buffer& buffer) {
    return width_split ? buffer_page_mapping->all_cores_
                       : corerange_to_cores(
                             buffer.shard_spec().grid(),
                             buffer.num_cores(),
                             buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR);
}

// Returns the host page to start reading from / writing to and the number of device pages to read from / write to
std::pair<uint32_t, uint32_t> calculate_pages_to_process_in_shard(
    uint32_t core_id,
    const Buffer& buffer,
    const std::shared_ptr<const BufferPageMapping>& buffer_page_mapping,
    uint32_t starting_host_page_idx,
    uint32_t ending_host_page_idx) {
    const std::vector<uint32_t> core_host_pages = buffer_page_mapping->core_host_page_indices_[core_id];
    TT_ASSERT(std::is_sorted(core_host_pages.begin(), core_host_pages.end()));

    auto is_host_page_within_region = [&](const uint32_t host_page) {
        return host_page >= starting_host_page_idx && host_page < ending_host_page_idx;
    };

    auto core_start_host_page_it =
        std::find_if(core_host_pages.begin(), core_host_pages.end(), is_host_page_within_region);
    auto core_end_host_page_it =
        std::find_if(core_host_pages.rbegin(), core_host_pages.rend(), is_host_page_within_region);

    // If we don't find a host page that lies at the start of the given region, we shouldn't find a host page that lies
    // at the end of it either
    TT_ASSERT((core_start_host_page_it == core_host_pages.end()) == (core_end_host_page_it == core_host_pages.rend()));

    const bool all_core_host_pages_outside_of_region = core_start_host_page_it == core_host_pages.end();
    if (all_core_host_pages_outside_of_region) {
        return {0, 0};
    }

    const uint32_t start_host_page = *(core_start_host_page_it);
    const uint32_t end_host_page = *(core_end_host_page_it);
    TT_ASSERT(end_host_page >= start_host_page);

    uint32_t num_dev_pages_to_process;

    const bool is_core_end_host_page_last_page_in_shard = core_end_host_page_it == core_host_pages.rbegin();
    if (is_core_end_host_page_last_page_in_shard) {
        const uint32_t num_dev_pages_in_shard =
            buffer_page_mapping->core_shard_shape_[core_id][0] * buffer.shard_spec().shape_in_pages()[1];
        num_dev_pages_to_process =
            num_dev_pages_in_shard - buffer_page_mapping->host_page_to_local_shard_page_mapping_[start_host_page];
    } else {
        const uint32_t host_page_after_end_host_page = *(core_end_host_page_it - 1);
        num_dev_pages_to_process =
            buffer_page_mapping->host_page_to_local_shard_page_mapping_[host_page_after_end_host_page] -
            buffer_page_mapping->host_page_to_local_shard_page_mapping_[start_host_page];
    }
    TT_ASSERT(num_dev_pages_to_process > 0);

    return {start_host_page, num_dev_pages_to_process};
}

void write_sharded_buffer_to_core(
    const void* src,
    uint32_t core_id,
    Buffer& buffer,
    ShardedBufferWriteDispatchParams& dispatch_params,
    const BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const CoreCoord core,
    CoreType dispatch_core_type) {
    // Skip writing the padded pages along the bottom
    // Currently since writing sharded tensors uses write_linear, we write the padded pages on width
    // Alternative write each page row into separate commands, or have a strided linear write
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t num_pages = 0;
    uint32_t remaining_pages_in_shard = dispatch_params.max_pages_per_shard;
    uint32_t curr_page_idx_in_shard = 0;
    if (dispatch_params.width_split) {
        const uint32_t ending_dst_host_page_index = dispatch_params.starting_dst_host_page_index +
                                                    dispatch_params.total_pages_written +
                                                    dispatch_params.total_pages_to_write;
        auto [host_page, num_pages_to_write] = calculate_pages_to_process_in_shard(
            core_id,
            buffer,
            dispatch_params.buffer_page_mapping,
            dispatch_params.starting_dst_host_page_index,
            ending_dst_host_page_index);
        num_pages = num_pages_to_write;

        if (num_pages == 0) {
            return;
        }

        dispatch_params.dst_page_index = dispatch_params.buffer_page_mapping->host_page_to_dev_page_mapping_[host_page];
        curr_page_idx_in_shard = dispatch_params.buffer_page_mapping->host_page_to_local_shard_page_mapping_[host_page];
        remaining_pages_in_shard -= curr_page_idx_in_shard;
    } else {
        while (remaining_pages_in_shard > 0 &&
               dispatch_params.initial_pages_skipped < dispatch_params.starting_dst_host_page_index) {
            dispatch_params.initial_pages_skipped += 1;
            curr_page_idx_in_shard += 1;
            remaining_pages_in_shard -= 1;
        }
        num_pages = std::min(dispatch_params.total_pages_to_write, remaining_pages_in_shard);
    }

    uint32_t bank_base_address = buffer.address();
    if (buffer.is_dram()) {
        bank_base_address += buffer.device()->allocator()->get_bank_offset(
            BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(core));
    }

    while (num_pages != 0) {
        // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
        uint32_t data_offset_bytes = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd));
        dispatch_params.issue_wait =
            dispatch_params.total_pages_written == 0;  // only stall for the first write of the buffer
        if (dispatch_params.issue_wait) {
            // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            data_offset_bytes *= 2;
        }
        uint32_t space_available_bytes = std::min(
            buf_dispatch_constants.issue_queue_cmd_limit -
                sysmem_manager.get_issue_queue_write_ptr(dispatch_params.cq_id),
            buf_dispatch_constants.max_prefetch_cmd_size);
        int32_t num_pages_available =
            (int32_t(space_available_bytes) - int32_t(data_offset_bytes)) / int32_t(dispatch_params.page_size_to_write);

        if (num_pages_available <= 0) {
            sysmem_manager.wrap_issue_queue_wr_ptr(dispatch_params.cq_id);
            continue;
        }

        dispatch_params.pages_per_txn = std::min(num_pages, (uint32_t)num_pages_available);
        dispatch_params.address = bank_base_address + curr_page_idx_in_shard * dispatch_params.page_size_to_write;
        dispatch_params.core = core;

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", dispatch_params.cq_id);

        issue_buffer_dispatch_command_sequence(src, buffer, dispatch_params, sub_device_ids, dispatch_core_type);
        curr_page_idx_in_shard += dispatch_params.pages_per_txn;
        num_pages -= dispatch_params.pages_per_txn;
        remaining_pages_in_shard -= dispatch_params.pages_per_txn;
        dispatch_params.dst_page_index += dispatch_params.pages_per_txn;
        dispatch_params.total_pages_to_write -= dispatch_params.pages_per_txn;
        dispatch_params.total_pages_written += dispatch_params.pages_per_txn;
    }
}

void validate_buffer_region_conditions(const Buffer& buffer, const BufferRegion& region) {
    TT_FATAL(
        buffer.is_valid_region(region),
        "Buffer region with offset {} and size {} is invalid.",
        region.offset,
        region.size);
    if (buffer.is_valid_partial_region(region)) {
        TT_FATAL(
            region.offset % buffer.page_size() == 0,
            "Offset {} must be divisible by the buffer page size {}.",
            region.offset,
            buffer.page_size());
        TT_FATAL(
            region.size % buffer.page_size() == 0,
            "Size {} must be divisible by the buffer page size {}.",
            region.size,
            buffer.page_size());
        TT_FATAL(
            (region.size + region.offset) <= buffer.size(),
            "(Size + offset) {} must be <= the buffer size {}.",
            region.size + region.offset,
            buffer.size());
    }
}

// Main API to write buffer data
void write_to_device_buffer(
    const void* src,
    Buffer& buffer,
    const BufferRegion& region,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    CoreType dispatch_core_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    validate_buffer_region_conditions(buffer, region);

    SystemMemoryManager& sysmem_manager = buffer.device()->sysmem_manager();
    const BufferDispatchConstants buf_dispatch_constants =
        generate_buffer_dispatch_constants(sysmem_manager, dispatch_core_type, cq_id);

    if (is_sharded(buffer.buffer_layout())) {
        ShardedBufferWriteDispatchParams dispatch_params = initialize_sharded_buf_dispatch_params(
            buffer, cq_id, expected_num_workers_completed, buf_dispatch_constants, region);
        const auto cores =
            get_cores_for_sharded_buffer(dispatch_params.width_split, dispatch_params.buffer_page_mapping, buffer);
        // Since we read core by core we are reading the device pages sequentially
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            write_sharded_buffer_to_core(
                src,
                core_id,
                buffer,
                dispatch_params,
                buf_dispatch_constants,
                sub_device_ids,
                cores[core_id],
                dispatch_core_type);
        }
    } else {
        std::unique_ptr<InterleavedBufferWriteDispatchParams> dispatch_params =
            initialize_interleaved_buf_dispatch_params(
                buffer, buf_dispatch_constants, cq_id, expected_num_workers_completed, region);
        write_interleaved_buffer_to_device(
            src, *dispatch_params, buffer, buf_dispatch_constants, sub_device_ids, dispatch_core_type);
    }
}

// ====== Utility Functions for Reads ======

// Initialize Dispatch Parameters - reused across write txns
ShardedBufferReadDispatchParams initialize_sharded_buf_read_dispatch_params(
    Buffer& buffer,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferRegion& region) {
    validate_buffer_region_conditions(buffer, region);

    // Note that the src_page_index is the device page idx, not the host page idx
    // Since we read core by core we are reading the device pages sequentially
    ShardedBufferReadDispatchParams dispatch_params;
    dispatch_params.cq_id = cq_id;
    dispatch_params.device = buffer.device();
    dispatch_params.padded_page_size = buffer.aligned_page_size();
    dispatch_params.initial_pages_skipped = 0;
    dispatch_params.src_page_index = region.offset / buffer.page_size();
    dispatch_params.starting_src_host_page_index = region.offset / buffer.page_size();
    dispatch_params.unpadded_dst_offset = 0;
    dispatch_params.width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
    dispatch_params.buffer_page_mapping = (dispatch_params.width_split) ? buffer.get_buffer_page_mapping() : nullptr;
    dispatch_params.total_pages_to_read = region.size / buffer.page_size();
    dispatch_params.total_pages_read = 0;
    dispatch_params.max_pages_per_shard = buffer.shard_spec().size();
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;
    return dispatch_params;
}

BufferReadDispatchParams initialize_interleaved_buf_read_dispatch_params(
    Buffer& buffer,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferRegion& region) {
    validate_buffer_region_conditions(buffer, region);

    BufferReadDispatchParams dispatch_params;
    dispatch_params.pages_per_txn = region.size / buffer.page_size();
    dispatch_params.src_page_index = region.offset / buffer.page_size();
    dispatch_params.cq_id = cq_id;
    dispatch_params.device = buffer.device();
    dispatch_params.padded_page_size = buffer.aligned_page_size();
    dispatch_params.unpadded_dst_offset = 0;
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;
    return dispatch_params;
}

// Issue dispatch commands for forwarding device buffer data to the Completion Queue
template <typename T>
void issue_read_buffer_dispatch_command_sequence(
    Buffer& buffer, T& dispatch_params, tt::stl::Span<const SubDeviceId> sub_device_ids, CoreType dispatch_core_type) {
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t num_worker_counters = sub_device_ids.size();
    // accounts for padding
    uint32_t cmd_sequence_sizeB =
        hal.get_alignment(HalMemType::HOST) *
            num_worker_counters +              // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        hal.get_alignment(HalMemType::HOST) +  // CQ_PREFETCH_CMD_STALL
        hal.get_alignment(
            HalMemType::HOST) +  // CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST
        hal.get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_RELAY_LINEAR or CQ_PREFETCH_CMD_RELAY_PAGED

    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    uint32_t dispatch_message_base_addr =
        DispatchMemMap::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    uint32_t last_index = num_worker_counters - 1;
    // We only need the write barrier + prefetch stall for the last wait cmd
    for (uint32_t i = 0; i < last_index; ++i) {
        auto offset_index = sub_device_ids[i].to_index();
        uint32_t dispatch_message_addr =
            dispatch_message_base_addr +
            DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
        command_sequence.add_dispatch_wait(
            false, dispatch_message_addr, dispatch_params.expected_num_workers_completed[offset_index]);
    }
    auto offset_index = sub_device_ids[last_index].to_index();
    uint32_t dispatch_message_addr =
        dispatch_message_base_addr + DispatchMemMap::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
    command_sequence.add_dispatch_wait_with_prefetch_stall(
        true, dispatch_message_addr, dispatch_params.expected_num_workers_completed[offset_index]);

    bool flush_prefetch = false;
    command_sequence.add_dispatch_write_host(
        flush_prefetch, dispatch_params.pages_per_txn * dispatch_params.padded_page_size, false);

    // Buffer layout specific logic
    if constexpr (std::is_same_v<T, ShardedBufferReadDispatchParams>) {
        const CoreCoord virtual_core =
            buffer.device()->virtual_core_from_logical_core(dispatch_params.core, buffer.core_type());
        command_sequence.add_prefetch_relay_linear(
            dispatch_params.device->get_noc_unicast_encoding(dispatch_downstream_noc, virtual_core),
            dispatch_params.padded_page_size * dispatch_params.pages_per_txn,
            dispatch_params.address);
    } else {
        command_sequence.add_prefetch_relay_paged(
            buffer.is_dram(),
            dispatch_params.src_page_index,
            dispatch_params.address,
            dispatch_params.padded_page_size,
            dispatch_params.pages_per_txn);
    }

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
}

// Top level functions to copy device buffers into the completion queue
void copy_sharded_buffer_from_core_to_completion_queue(
    uint32_t core_id,
    Buffer& buffer,
    ShardedBufferReadDispatchParams& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const CoreCoord core,
    CoreType dispatch_core_type) {
    uint32_t pages_per_txn = 0;
    uint32_t curr_page_idx_in_shard = 0;
    uint32_t host_page = 0;
    uint32_t address = buffer.address();

    if (dispatch_params.width_split) {
        const uint32_t ending_src_host_page_index = dispatch_params.starting_src_host_page_index +
                                                    dispatch_params.total_pages_read +
                                                    dispatch_params.total_pages_to_read;
        auto [start_host_page, num_pages_to_read] = calculate_pages_to_process_in_shard(
            core_id,
            buffer,
            dispatch_params.buffer_page_mapping,
            dispatch_params.starting_src_host_page_index,
            ending_src_host_page_index);
        host_page = start_host_page;
        pages_per_txn = num_pages_to_read;
        if (pages_per_txn > 0) {
            dispatch_params.src_page_index =
                dispatch_params.buffer_page_mapping->host_page_to_dev_page_mapping_[host_page];
            curr_page_idx_in_shard =
                dispatch_params.buffer_page_mapping->host_page_to_local_shard_page_mapping_[host_page];
        }
    } else {
        host_page = dispatch_params.src_page_index;
        pages_per_txn = std::min(dispatch_params.total_pages_to_read, dispatch_params.max_pages_per_shard);

        if (dispatch_params.initial_pages_skipped + dispatch_params.max_pages_per_shard <=
            dispatch_params.starting_src_host_page_index) {
            pages_per_txn = 0;
            dispatch_params.initial_pages_skipped += dispatch_params.max_pages_per_shard;
        } else if (core_id == dispatch_params.starting_src_host_page_index / dispatch_params.max_pages_per_shard) {
            dispatch_params.initial_pages_skipped +=
                (dispatch_params.starting_src_host_page_index - dispatch_params.initial_pages_skipped);
            const uint32_t remaining_pages_in_shard =
                ((core_id + 1) * dispatch_params.max_pages_per_shard) - dispatch_params.initial_pages_skipped;
            curr_page_idx_in_shard = dispatch_params.max_pages_per_shard - remaining_pages_in_shard;
            pages_per_txn = std::min(pages_per_txn, remaining_pages_in_shard);
        }
    }

    if (buffer.is_dram()) {
        address += buffer.device()->allocator()->get_bank_offset(
            BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(core));
    }
    address += curr_page_idx_in_shard * buffer.aligned_page_size();

    dispatch_params.total_pages_to_read -= pages_per_txn;
    dispatch_params.total_pages_read += pages_per_txn;
    dispatch_params.pages_per_txn = pages_per_txn;

    if (dispatch_params.pages_per_txn > 0) {
        dispatch_params.unpadded_dst_offset =
            (host_page - dispatch_params.starting_src_host_page_index) * buffer.page_size();
        dispatch_params.address = address;
        dispatch_params.core = core;
        issue_read_buffer_dispatch_command_sequence(buffer, dispatch_params, sub_device_ids, dispatch_core_type);
    }
}

void copy_interleaved_buffer_to_completion_queue(
    BufferReadDispatchParams& dispatch_params,
    Buffer& buffer,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type) {
    if (dispatch_params.pages_per_txn > 0) {
        uint32_t bank_base_address = buffer.address();

        // Only 8 bits are assigned for the page offset in CQPrefetchRelayPagedCmd
        // To handle larger page offsets move bank base address up and update page offset to be relative to the new
        // bank address
        if (dispatch_params.src_page_index > CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK) {
            const uint32_t num_banks = dispatch_params.device->allocator()->get_num_banks(buffer.buffer_type());
            const uint32_t num_pages_per_bank = dispatch_params.src_page_index / num_banks;
            bank_base_address += num_pages_per_bank * buffer.aligned_page_size();
            dispatch_params.src_page_index = dispatch_params.src_page_index % num_banks;
        }
        dispatch_params.address = bank_base_address;
        issue_read_buffer_dispatch_command_sequence(buffer, dispatch_params, sub_device_ids, dispatch_core_type);
    }
}

// Functions used to copy buffer data from completion queue into user space
std::shared_ptr<tt::tt_metal::CompletionReaderVariant> generate_sharded_buffer_read_descriptor(
    void* dst, ShardedBufferReadDispatchParams& dispatch_params, Buffer& buffer) {
    // Increment the src_page_index after the Read Buffer Descriptor has been populated
    // for the current core/txn
    auto initial_src_page_index = dispatch_params.src_page_index;
    dispatch_params.src_page_index += dispatch_params.pages_per_txn;
    return std::make_shared<tt::tt_metal::CompletionReaderVariant>(
        std::in_place_type<tt::tt_metal::ReadBufferDescriptor>,
        buffer.buffer_layout(),
        buffer.page_size(),
        dispatch_params.padded_page_size,
        dst,
        dispatch_params.unpadded_dst_offset,
        dispatch_params.pages_per_txn,
        initial_src_page_index,
        dispatch_params.starting_src_host_page_index,
        dispatch_params.buffer_page_mapping);
}

std::shared_ptr<tt::tt_metal::CompletionReaderVariant> generate_interleaved_buffer_read_descriptor(
    void* dst, BufferReadDispatchParams& dispatch_params, Buffer& buffer) {
    return std::make_shared<tt::tt_metal::CompletionReaderVariant>(
        std::in_place_type<tt::tt_metal::ReadBufferDescriptor>,
        buffer.buffer_layout(),
        buffer.page_size(),
        dispatch_params.padded_page_size,
        dst,
        dispatch_params.unpadded_dst_offset,
        dispatch_params.pages_per_txn,
        dispatch_params.src_page_index);
}

void copy_completion_queue_data_into_user_space(
    const ReadBufferDescriptor& read_buffer_descriptor,
    chip_id_t mmio_device_id,
    uint16_t channel,
    uint32_t cq_id,
    SystemMemoryManager& sysmem_manager,
    volatile bool& exit_condition) {
    const auto& [buffer_layout, page_size, padded_page_size, buffer_page_mapping, dst, dst_offset, num_pages_read, cur_dev_page_id, starting_host_page_id] =
        read_buffer_descriptor;
    uint32_t padded_num_bytes = (num_pages_read * padded_page_size) + sizeof(CQDispatchCmd);
    uint32_t contig_dst_offset = dst_offset;
    uint32_t remaining_bytes_to_read = padded_num_bytes;
    uint32_t dev_page_id = cur_dev_page_id;

    // track the amount of bytes read in the last non-aligned page
    uint32_t remaining_bytes_of_nonaligned_page = 0;
    std::optional<uint32_t> host_page_id = std::nullopt;
    uint32_t offset_in_completion_q_data = sizeof(CQDispatchCmd);

    uint32_t pad_size_bytes = padded_page_size - page_size;

    while (remaining_bytes_to_read != 0) {
        uint32_t completion_queue_write_ptr_and_toggle =
            sysmem_manager.completion_queue_wait_front(cq_id, exit_condition);

        if (exit_condition) {
            break;
        }

        uint32_t completion_q_write_ptr = (completion_queue_write_ptr_and_toggle & 0x7fffffff) << 4;
        uint32_t completion_q_write_toggle = completion_queue_write_ptr_and_toggle >> (31);
        uint32_t completion_q_read_ptr = sysmem_manager.get_completion_queue_read_ptr(cq_id);
        uint32_t completion_q_read_toggle = sysmem_manager.get_completion_queue_read_toggle(cq_id);

        uint32_t bytes_avail_in_completion_queue;
        if (completion_q_write_ptr > completion_q_read_ptr and completion_q_write_toggle == completion_q_read_toggle) {
            bytes_avail_in_completion_queue = completion_q_write_ptr - completion_q_read_ptr;
        } else {
            // Completion queue write pointer on device wrapped but read pointer is lagging behind.
            //  In this case read up until the end of the completion queue first
            bytes_avail_in_completion_queue = sysmem_manager.get_completion_queue_limit(cq_id) - completion_q_read_ptr;
        }

        // completion queue write ptr on device could have wrapped but our read ptr is lagging behind
        uint32_t bytes_xfered = std::min(remaining_bytes_to_read, bytes_avail_in_completion_queue);
        uint32_t num_pages_xfered = div_up(bytes_xfered, DispatchSettings::TRANSFER_PAGE_SIZE);

        remaining_bytes_to_read -= bytes_xfered;

        if (buffer_page_mapping == nullptr) {
            void* contiguous_dst = (void*)(uint64_t(dst) + contig_dst_offset);
            if (page_size == padded_page_size) {
                uint32_t data_bytes_xfered = bytes_xfered - offset_in_completion_q_data;
                tt::Cluster::instance().read_sysmem(
                    contiguous_dst,
                    data_bytes_xfered,
                    completion_q_read_ptr + offset_in_completion_q_data,
                    mmio_device_id,
                    channel);
                contig_dst_offset += data_bytes_xfered;
                offset_in_completion_q_data = 0;
            } else {
                uint32_t src_offset_bytes = offset_in_completion_q_data;
                offset_in_completion_q_data = 0;
                uint32_t dst_offset_bytes = 0;

                while (src_offset_bytes < bytes_xfered) {
                    uint32_t src_offset_increment = padded_page_size;
                    uint32_t num_bytes_to_copy;
                    if (remaining_bytes_of_nonaligned_page > 0) {
                        // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                        remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                        src_offset_increment = num_bytes_to_copy;
                        // We finished copying the page
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                            // There is more data after padding
                            if (rem_bytes_in_cq >= pad_size_bytes) {
                                src_offset_increment += pad_size_bytes;
                                // Only pad data left in queue
                            } else {
                                offset_in_completion_q_data = pad_size_bytes - rem_bytes_in_cq;
                            }
                        }
                    } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                        // Case 2: Last page of data that was popped off the completion queue
                        // Don't need to compute src_offset_increment since this is end of loop
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                        remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                        // We've copied needed data, start of next read is offset due to remaining pad bytes
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        }
                    } else {
                        num_bytes_to_copy = page_size;
                    }

                    tt::Cluster::instance().read_sysmem(
                        (char*)(uint64_t(contiguous_dst) + dst_offset_bytes),
                        num_bytes_to_copy,
                        completion_q_read_ptr + src_offset_bytes,
                        mmio_device_id,
                        channel);

                    src_offset_bytes += src_offset_increment;
                    dst_offset_bytes += num_bytes_to_copy;
                    contig_dst_offset += num_bytes_to_copy;
                }
            }
        } else {
            uint32_t src_offset_bytes = offset_in_completion_q_data;
            offset_in_completion_q_data = 0;
            uint32_t dst_offset_bytes = contig_dst_offset;
            uint32_t num_bytes_to_copy = 0;

            while (src_offset_bytes < bytes_xfered) {
                uint32_t src_offset_increment = padded_page_size;
                if (remaining_bytes_of_nonaligned_page > 0) {
                    // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                    remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                    src_offset_increment = num_bytes_to_copy;
                    // We finished copying the page
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        dev_page_id++;
                        uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                        // There is more data after padding
                        if (rem_bytes_in_cq >= pad_size_bytes) {
                            src_offset_increment += pad_size_bytes;
                            offset_in_completion_q_data = 0;
                            // Only pad data left in queue
                        } else {
                            offset_in_completion_q_data = (pad_size_bytes - rem_bytes_in_cq);
                        }
                    }
                    if (!host_page_id.has_value()) {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                    // Case 2: Last page of data that was popped off the completion queue
                    // Don't need to compute src_offset_increment since this is end of loop
                    host_page_id = buffer_page_mapping->dev_page_to_host_page_mapping_[dev_page_id];
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                    remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                    // We've copied needed data, start of next read is offset due to remaining pad bytes
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        dev_page_id++;
                    }
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = (*host_page_id - starting_host_page_id) * page_size;
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else {
                    num_bytes_to_copy = page_size;
                    host_page_id = buffer_page_mapping->dev_page_to_host_page_mapping_[dev_page_id];
                    dev_page_id++;
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = (*host_page_id - starting_host_page_id) * page_size;
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                }

                tt::Cluster::instance().read_sysmem(
                    (char*)(uint64_t(dst) + dst_offset_bytes),
                    num_bytes_to_copy,
                    completion_q_read_ptr + src_offset_bytes,
                    mmio_device_id,
                    channel);

                src_offset_bytes += src_offset_increment;
            }
            dst_offset_bytes += num_bytes_to_copy;
            contig_dst_offset = dst_offset_bytes;
        }
        sysmem_manager.completion_queue_pop_front(num_pages_xfered, cq_id);
    }
}

tt::stl::Span<const SubDeviceId> select_sub_device_ids(
    IDevice* device, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    if (sub_device_ids.empty()) {
        return device->get_sub_device_stall_group();
    } else {
        for (const auto& sub_device_id : sub_device_ids) {
            TT_FATAL(
                sub_device_id.to_index() < device->num_sub_devices(),
                "Invalid sub-device id specified {}",
                sub_device_id.to_index());
        }
        return sub_device_ids;
    }
}

template void issue_buffer_dispatch_command_sequence<InterleavedBufferWriteDispatchParams>(
    const void*, Buffer&, InterleavedBufferWriteDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);
template void issue_buffer_dispatch_command_sequence<ShardedBufferWriteDispatchParams>(
    const void*, Buffer&, ShardedBufferWriteDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);

template void issue_read_buffer_dispatch_command_sequence<BufferReadDispatchParams>(
    Buffer&, BufferReadDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);
template void issue_read_buffer_dispatch_command_sequence<ShardedBufferReadDispatchParams>(
    Buffer&, ShardedBufferReadDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);

}  // namespace buffer_dispatch

}  // namespace tt::tt_metal
