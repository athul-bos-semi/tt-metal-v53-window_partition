// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
#include <filesystem>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "llrt/llrt.hpp"
#include "tools/profiler/profiler_state.hpp"
#include "tools/profiler/common.hpp"
#include "tools/profiler/profiler_optional_metadata.hpp"
#include "tracy/TracyTTDevice.hpp"
#include "common/TracyTTDeviceData.hpp"

#include <nlohmann/json.hpp>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;

namespace tt {

namespace tt_metal {

class DeviceProfiler {
private:
    // Device architecture
    tt::ARCH device_architecture;

    // Device frequency
    int device_core_frequency;

    // Smallest timestamp
    uint64_t smallest_timestamp = (1lu << 63);

    // Output Dir for device Profile Logs
    std::filesystem::path output_dir;

    // Device-Core tracy context
    std::map<std::pair<uint16_t, CoreCoord>, TracyTTCtx> device_tracy_contexts;

    // Device events
    std::set<tracy::TTDeviceEvent> device_events;

    // Hash to zone source locations
    std::unordered_map<uint16_t, std::string> hash_to_zone_src_locations;

    // Zone sourece locations
    std::unordered_set<std::string> zone_src_locations;

    // 32bit FNV-1a hashing
    uint32_t hash32CT(const char* str, size_t n, uint32_t basis = UINT32_C(2166136261));

    // XORe'd 16-bit FNV-1a hashing functions
    uint16_t hash16CT(const std::string& str);

    // Iterate through all zone source locations and generate hash
    void generateZoneSourceLocationsHashes();

    // serialize all noc trace data into per-op json trace files
    void serializeJsonNocTraces(
        const nlohmann::json& noc_trace_json_log,
        const std::filesystem::path& output_dir,
        int device_id,
        bool lastDump);

    void emitCSVHeader(
        std::ofstream& log_file_ofs, const tt::ARCH& device_architecture, int device_core_frequency) const;

    // translates potentially-virtual coordinates recorded on Device into physical coordinates
    CoreCoord getPhysicalAddressFromVirtual(const IDevice* device, const CoreCoord& c) const;

    // Dumping profile result to file
    void logPacketData(
        const IDevice* device,
        std::ofstream& log_file_ofs,
        nlohmann::json& noc_trace_json_log,
        uint32_t runID,
        uint32_t runHostID,
        std::string opname,
        int device_id,
        CoreCoord core,
        int core_flat,
        int risc_num,
        uint64_t stat_value,
        uint32_t timer_id,
        uint64_t timestamp);

    // logs packet data to CSV file
    void logPacketDataToCSV(
        const IDevice* device,
        std::ofstream& log_file_ofs,
        int device_id,
        int core_x,
        int core_y,
        const std::string_view risc_name,
        uint32_t timer_id,
        uint64_t timestamp,
        uint64_t data,
        uint32_t run_id,
        uint32_t run_host_id,
        const std::string_view opname,
        const std::string_view zone_name,
        kernel_profiler::PacketTypes packet_type,
        uint64_t source_line,
        const std::string_view source_file);

    // dump noc trace related profile data to json file
    void logNocTracePacketDataToJson(
        const IDevice* device,
        nlohmann::json& noc_trace_json_log,
        int device_id,
        int core_x,
        int core_y,
        const std::string_view risc_name,
        uint32_t timer_id,
        uint64_t timestamp,
        uint64_t data,
        uint32_t run_id,
        uint32_t run_host_id,
        const std::string_view opname,
        const std::string_view zone_name,
        kernel_profiler::PacketTypes packet_type,
        uint64_t source_line,
        const std::string_view source_file);

    // Helper function for reading risc profile results
    void readRiscProfilerResults(
        const IDevice* device,
        const std::optional<ProfilerOptionalMetadata>& metadata,
        std::ofstream& log_file_ofs,
        nlohmann::json& noc_trace_json_log,
        int device_id,
        const std::vector<std::uint32_t>& profile_buffer,
        const CoreCoord& worker_core);

    // Push device results to tracy
    void pushTracyDeviceResults();

    // Track the smallest timestamp dumped to file
    void firstTimestamp(uint64_t timestamp);

public:
    DeviceProfiler(const bool new_logs);

    DeviceProfiler() = delete;

    ~DeviceProfiler();

    // DRAM buffer for device side results
    std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer = nullptr;
    std::shared_ptr<tt::tt_metal::Program> sync_program = nullptr;

    // Device-core Syncdata
    std::map<CoreCoord, std::tuple<double, double, double>> device_core_sync_info;

    // Freshen device logs
    void freshDeviceLog();

    // Set the device architecture
    void setDeviceArchitecture(tt::ARCH device_arch);

    // Change the output dir of device profile logs
    void setOutputDir(const std::string& new_output_dir);

    // Traverse all cores on the device and dump the device profile results
    void dumpResults(
        IDevice* device,
        const std::vector<CoreCoord>& worker_cores,
        bool lastDump,
        const std::optional<ProfilerOptionalMetadata>& metadata = {});
};

}  // namespace tt_metal

}  // namespace tt
