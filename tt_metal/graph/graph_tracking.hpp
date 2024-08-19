// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <any>
#include <span>
#include <string_view>

#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/buffers/buffer.hpp"

namespace tt::tt_metal {

    class Program;
    class IGraphProcessor{
    public:
        enum class RunMode {
            REAL,
            FAKE
        };

        IGraphProcessor() = default;

        virtual void track_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) {};

        virtual void track_deallocate(tt::tt_metal::Buffer* buffer) {};

        virtual void track_allocate_cb(const CoreRangeSet &core_range_set, uint64_t addr, uint64_t size) {};

        virtual void track_deallocate_cb() {};

        virtual void track_program(tt::tt_metal::Program* program) {};

        virtual void track_begin_function(std::string_view function_name, std::span<std::any> input_parameters) {};

        virtual void track_end_function() {};
        virtual void track_end_function(const std::any& output_tensors) {};

        virtual void begin_capture(RunMode mode) {};

        virtual nlohmann::json end_capture() {return nullptr;};

        virtual ~IGraphProcessor() = default;

    };

    class IGraphHooks {
    public:
        IGraphHooks() = default;
        virtual bool hook_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) = 0;

        virtual bool hook_deallocate(tt::tt_metal::Buffer* buffer) = 0;

        virtual bool hook_program(Program* program) = 0;

        virtual ~IGraphHooks() = default;
    };

    class GraphTracker {
    public:
        static GraphTracker& instance() {
            static GraphTracker tracker;
            return tracker;
        }

        void push_processor(const std::shared_ptr<IGraphProcessor>& processor);
        void pop_processor();

        bool add_hook(const std::shared_ptr<IGraphHooks>& hook);

        void track_allocate(Buffer* buffer, bool bottom_up);

        void track_deallocate(Buffer* buffer);

        void track_allocate_cb(const CoreRangeSet &core_range_set, uint64_t addr, uint64_t size);

        void track_deallocate_cb();

        void track_program(Program* program);

        template<class... Args>
        void track_begin_function(std::string_view function_name, Args&&... args) {
            if (processors.empty()) {
                return;
            }
            std::array<std::any, sizeof...(Args)>  params{std::any(std::ref(args))...};
            for (auto& it : processors) {
                it->track_begin_function(function_name, params);
            }
        }

        // Track op that doesn't return anything
        void track_end_function() {
            if (processors.empty()) {
                return;
            }
            for (auto& it : processors) {
                it->track_end_function();
            }
        }

        template<class ReturnType>
        void track_end_function(ReturnType&& output_tensors) {
            if (processors.empty()) {
                return;
            }
            for (auto& it : processors) {
                it->track_end_function(std::ref(output_tensors));
            }
        }

        bool hook_allocate(Buffer* buffer, bool bottom_up);

        bool hook_deallocate(Buffer* buffer);

        bool hook_program(tt::tt_metal::Program* program);

        const std::vector<std::shared_ptr<IGraphProcessor>>& get_processors() const;

        const std::shared_ptr<IGraphHooks>& get_hooks() const;

        void clear();

       private:
        GraphTracker() = default;
        ~GraphTracker() = default;
        GraphTracker(const GraphTracker&) = delete;
        GraphTracker(GraphTracker&&) = delete;

        std::vector<std::shared_ptr<IGraphProcessor>> processors;

        std::shared_ptr<IGraphHooks> hook;

    };
}
