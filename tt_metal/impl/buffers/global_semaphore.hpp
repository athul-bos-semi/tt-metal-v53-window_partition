// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_metal {

inline namespace v0 {

class Buffer;
class Device;

class GlobalSemaphore {
public:
    GlobalSemaphore(
        Device* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    GlobalSemaphore(
        Device* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    GlobalSemaphore(const GlobalSemaphore&) = default;
    GlobalSemaphore& operator=(const GlobalSemaphore&) = default;

    GlobalSemaphore(GlobalSemaphore&&) noexcept = default;
    GlobalSemaphore& operator=(GlobalSemaphore&&) noexcept = default;

    static std::shared_ptr<GlobalSemaphore> create(
        Device* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    static std::shared_ptr<GlobalSemaphore> create(
        Device* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    Device* device() const;

    DeviceAddr address() const;

    void reset_semaphore_value();

    static constexpr auto attribute_names = std::forward_as_tuple("cores", "initial_value");
    const auto attribute_values() const { return std::make_tuple(this->cores_, this->initial_value_); }

private:
    void setup_buffer(BufferType buffer_type);

    // GlobalSemaphore is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    std::shared_ptr<Buffer> buffer_;
    std::vector<uint32_t> host_buffer_;
    Device* device_;
    CoreRangeSet cores_;
    uint32_t initial_value_ = 0;
};

}  // namespace v0

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::GlobalSemaphore> {
    std::size_t operator()(const tt::tt_metal::GlobalSemaphore& global_semaphore) const;
};

}  // namespace std
