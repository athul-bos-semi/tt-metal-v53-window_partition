// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dispatch_fixture.hpp"
#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <circular_buffer_constants.h>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "lightmetal/lightmetal_replay.hpp"
#include "command_queue_fixture.hpp"
#include <lightmetal_binary.hpp>

class SingleDeviceLightMetalFixture : public CommandQueueFixture {
protected:
    bool replay_binary_;
    std::string trace_bin_path_;
    bool write_bin_to_disk_;

    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }

    void CreateDeviceAndBeginCapture(
        const size_t trace_region_size, const bool replay_binary = true, const std::string trace_bin_path = "") {
        // Skip writing to disk by default, unless user sets env var for local testing
        write_bin_to_disk_ = tt::parse_env("LIGHTMETAL_SAVE_BINARY", false);

        // If user didn't provide a specific trace bin path, set a default here based on test name
        if (trace_bin_path == "") {
            const auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
            auto trace_filename = test_info ? std::string(test_info->name()) + ".bin" : "lightmetal_trace.bin";
            this->trace_bin_path_ = "/tmp/" + trace_filename;
        }

        this->create_device(trace_region_size);
        this->replay_binary_ = replay_binary && !tt::parse_env("LIGHTMETAL_DISABLE_RUN", false);
        // TODO (kmabee) - revisit placement. CreateDevice() path calls CreateKernel() on programs not
        // created with CreateProgram() traced API which leads to "program not in global_id map"
        LightMetalBeginCapture();
    }

    // End light metal tracing, write to optional filename and optionally run from binary blob
    void TearDown() override {
        LightMetalBinary binary = LightMetalEndCapture();

        if (binary.is_empty()) {
            FAIL() << "Light Metal Binary is empty for test, unexpected.";
        }
        if (write_bin_to_disk_ && !this->trace_bin_path_.empty() && !binary.is_empty()) {
            log_info(tt::LogTest, "Writing light metal binary {} bytes to {}", binary.size(), this->trace_bin_path_);
            binary.save_to_file(this->trace_bin_path_);
        }

        if (!this->IsSlowDispatch()) {
            tt::tt_metal::CloseDevice(this->device_);
        }

        // We could gaurd this to not attempt to replay empty binary, and still allow test to pass
        // but, would rather catch the case if the feature gets disabled at compile time.
        if (replay_binary_) {
            RunLightMetalBinary(std::move(binary));
        }
    }

    // Mimic the light-metal standalone run replay tool by executing the binary.
    void RunLightMetalBinary(LightMetalBinary&& binary) {
        tt::tt_metal::LightMetalReplay lm_replay(std::move(binary));
        if (!lm_replay.execute_binary()) {
            FAIL() << "Light Metal Binary failed to execute or encountered errors.";
        } else {
            log_info(tt::LogMetalTrace, "Light Metal Binary executed successfully!");
        }
    }
};
