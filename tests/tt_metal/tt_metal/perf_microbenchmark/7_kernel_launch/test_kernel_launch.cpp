// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////
// This test measures the time for executing a program that contains empty data
// movement kernels and compute kernel.
//
// Usage example:
//   ./test_kernel_launch --cores-r <number of cores in a row> --cores-c <number
//   of cores in a column> --core-gropus <number of core grops where each core
//   group executes different kernel binaries>
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
    log_fatal("Test not supported w/ slow dispatch, exiting");
  }

  bool pass = true;
  unsigned long elapsed_us;

  ////////////////////////////////////////////////////////////////////////////
  //                      Initial Runtime Args Parse
  ////////////////////////////////////////////////////////////////////////////
  std::vector<std::string> input_args(argv, argv + argc);
  uint32_t num_cores_r;
  uint32_t num_cores_c;
  uint32_t num_core_groups;
  try {
    std::tie(num_cores_r, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--cores-r", 9);
    std::tie(num_cores_c, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--cores-c", 12);

    std::tie(num_core_groups, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--core-groups", 4);
  } catch (const std::exception& e) {
    log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
  }

  if (num_cores_r < num_core_groups) {
    log_fatal(
        tt::LogTest,
        "The number of cores in a row ({}) must be bigger than or equal than "
        "the number of core groups ({})",
        num_cores_r, num_core_groups);
  }

  ////////////////////////////////////////////////////////////////////////////
  //                      Device Setup
  ////////////////////////////////////////////////////////////////////////////
  int device_id = 0;
  tt_metal::Device* device = tt_metal::CreateDevice(device_id);
  CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;

  try {
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();
    uint32_t single_tile_size = 2 * 1024;

    for (int core_group_idx = 0; core_group_idx < num_core_groups;
         ++core_group_idx) {
      CoreCoord start_core = {0,
                              (num_cores_r / num_core_groups) * core_group_idx};
      CoreCoord end_core = {
          (std::size_t)num_cores_c - 1,
          (core_group_idx == num_core_groups - 1)
              ? (std::size_t)num_cores_r - 1
              : (num_cores_r / num_core_groups) * (core_group_idx + 1) - 1};
      CoreRange group_of_cores{.start = start_core, .end = end_core};

      log_info(
          LogTest, "Setting kernels for core group {}, cores ({},{}) ~ ({},{})",
          core_group_idx, start_core.x, start_core.y, end_core.x, end_core.y);

      for (int i = start_core.y; i <= end_core.y; i++) {
        for (int j = start_core.x; j <= end_core.x; j++) {
          CoreCoord core = {(std::size_t)j, (std::size_t)i};
          uint32_t cb_index = 0;
          uint32_t cb_tiles = 8;
          auto cb_src0 = tt_metal::CreateCircularBuffer(
              program, cb_index, core, cb_tiles, cb_tiles * single_tile_size,
              tt::DataFormat::Float16_b);
        }
      }

      vector<uint32_t> reader_compile_args = {uint32_t(core_group_idx)};
      auto reader_kernel = tt_metal::CreateDataMovementKernel(
          program,
          "tests/tt_metal/tt_metal/perf_microbenchmark/7_kernel_launch/"
          "kernels/"
          "reader.cpp",
          group_of_cores,
          tt_metal::DataMovementConfig{
              .processor = tt_metal::DataMovementProcessor::RISCV_1,
              .noc = tt_metal::NOC::RISCV_1_default,
              .compile_args = reader_compile_args});

      vector<uint32_t> writer_compile_args = {uint32_t(core_group_idx)};
      auto writer_kernel = tt_metal::CreateDataMovementKernel(
          program,
          "tests/tt_metal/tt_metal/perf_microbenchmark/7_kernel_launch/"
          "kernels/"
          "writer.cpp",
          group_of_cores,
          tt_metal::DataMovementConfig{
              .processor = tt_metal::DataMovementProcessor::RISCV_0,
              .noc = tt_metal::NOC::RISCV_0_default,
              .compile_args = writer_compile_args});

      vector<uint32_t> compute_compile_args = {uint32_t(core_group_idx)};
      auto compute_kernel = tt_metal::CreateComputeKernel(
          program,
          "tests/tt_metal/tt_metal/perf_microbenchmark/7_kernel_launch/"
          "kernels/"
          "compute.cpp",
          group_of_cores,
          tt_metal::ComputeConfig{.compile_args = compute_compile_args});

      for (int i = start_core.y; i <= end_core.y; i++) {
        for (int j = start_core.x; j <= end_core.x; j++) {
          CoreCoord core = {(std::size_t)j, (std::size_t)i};
          int core_index = i * num_cores_c + j;

          vector<uint32_t> reader_runtime_args;
          vector<uint32_t> writer_runtime_args;
          for (uint32_t k = 0; k < 255; ++k) {
            reader_runtime_args.push_back(core_index + k);
            writer_runtime_args.push_back(core_index + k);
          }

          SetRuntimeArgs(program, writer_kernel, core, writer_runtime_args);
          SetRuntimeArgs(program, reader_kernel, core, reader_runtime_args);
        }
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::detail::CompileProgram(device, program);

    auto t_begin = std::chrono::steady_clock::now();
    EnqueueProgram(cq, program, false);
    Finish(cq);
    auto t_end = std::chrono::steady_clock::now();
    elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();

    log_info(LogTest, "Time elapsed for executing empty kernels: {}us",
             elapsed_us);

    pass &= tt_metal::CloseDevice(device);
  } catch (const std::exception& e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
    log_error(LogTest, "System error message: {}", std::strerror(errno));
  }

  // Determine if it passes performance goal
  if (pass) {
    // goal is under 10us
    long target_us = 10;

    if (elapsed_us > target_us) {
      pass = false;
      log_error(LogTest,
                "The kernel launch overhead does not meet the criteria. "
                "Current: {}us, goal: <{}us",
                elapsed_us, target_us);
    }
  }

  if (pass) {
    log_info(LogTest, "Test Passed");
  } else {
    log_fatal(LogTest, "Test Failed");
  }

  TT_ASSERT(pass);

  return 0;
}
