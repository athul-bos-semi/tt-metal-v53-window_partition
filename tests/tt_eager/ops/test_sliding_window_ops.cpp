// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tuple>

#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/reference_sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/types.hpp"

using std::vector;
using tt::tt_metal::Tensor;
using namespace ttnn::operations::sliding_window;

// From owned_buffer of type bfloat16 of create float vector for convolution operation.
vector<float> create_filter_vec(
    const owned_buffer::Buffer<bfloat16>& filter_tensor_buf, uint32_t filter_h, uint32_t filter_w) {
    vector<float> filter_vector;
    for (auto h = 0; h < filter_h; h++) {
        for (auto w = 0; w < filter_w; w++) {
            filter_vector.push_back(filter_tensor_buf[h * filter_w + w].to_float());
        }
    }
    return filter_vector;
}

// Compare calculated convolution buffer with Golden convolution
uint32_t compare_conv_out_with_golden(
    const owned_buffer::Buffer<bfloat16>& out_golden_tensor_buf,
    const owned_buffer::Buffer<bfloat16>& conv_tensor_buf) {
    uint32_t diff = 0;
    if (out_golden_tensor_buf != conv_tensor_buf) {
        assert(out_golden_tensor_buf.size() == conv_tensor_buf.size());
        for (uint32_t i = 0; i < out_golden_tensor_buf.size(); i++) {
            if (out_golden_tensor_buf[i] != conv_tensor_buf[i]) {
                log_info(
                    tt::LogTest,
                    "Error at i = {}, Golden = {}, Calculated = {}",
                    i,
                    out_golden_tensor_buf[i].to_float(),
                    conv_tensor_buf[i].to_float());
                diff++;
            }
        }
    }
    return diff;
}

// Validate Flattened_* configs generated by generate_halo_kernel_config_tensors using pad_metadata.
// It is ok to use pad_metadata since its correctness is validated in other test cases.
uint32_t validate_generate_halo_kernel_config(
    tt::tt_metal::IDevice* device,
    const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries,
    const std::tuple<vector<vector<uint16_t>>, std::vector<std::vector<uint16_t>>, std::vector<std::vector<uint16_t>>>&
        halo_kernel_config,
    const vector<bool>& pad_metadata,
    bool remote_read = false,
    bool is_block_sharded = false,
    bool transpose_mcast = false) {
    auto [flattened_pad_config, flattened_local_config, flattened_remote_config] = halo_kernel_config;

    uint32_t padded_input_tensor_buf_idx = 0;
    uint32_t invalid_pads = 0, invalid_indices = 0;
    uint32_t failed_tests = 0;

    auto find_invalids = [&](vector<uint32_t>& indices, bool val) -> int {
        auto invalids = 0;
        for (auto idx : indices) {
            if (pad_metadata[idx] != val) {
                invalids++;
                log_info(
                    tt::LogTest,
                    "Error at index = {}, Expected = {}, Calculated = {}",
                    idx,
                    val,
                    bool(pad_metadata[idx]));
            }
        }
        return invalids;
    };

    auto pad_indices = pad_indices_from_flattened_pad_config(flattened_pad_config, shard_boundaries);
    invalid_pads = find_invalids(pad_indices, true);
    if (invalid_pads != 0) {
        log_error(
            tt::LogTest,
            "Failed to validate flattened_pad_config of halo_kernel_config, invalid pads = {}",
            invalid_pads);
        failed_tests++;
    }

    auto local_indices = input_indices_from_flattened_local_config(flattened_local_config, shard_boundaries);
    invalid_indices = find_invalids(local_indices, false);
    if (invalid_indices != 0) {
        log_error(
            tt::LogTest,
            "Failed to validate flattened_local_config of halo_kernel_config, invalid indices = {}",
            invalid_indices);
        failed_tests++;
    }
    auto remote_indices = input_indices_from_flattened_remote_config(
        device, flattened_remote_config, shard_boundaries, remote_read, is_block_sharded, transpose_mcast);
    invalid_indices = find_invalids(remote_indices, false);
    if (invalid_indices != 0) {
        log_error(
            tt::LogTest,
            "Failed to validate flattened_remote_config of halo_kernel_config, invalid indices = {}",
            invalid_indices);
        failed_tests++;
    }

    return failed_tests;
}

// Validate
// 1) various generate_* functions using reference convolution and convolution Calculated
// using outputs of these generate functions.
// 2) halo kernel configs using pad_metadata
uint32_t validate_generate_functions(
    tt::tt_metal::IDevice* device,
    const SlidingWindowConfig& config,
    const owned_buffer::Buffer<bfloat16>& input_padded_tensor_buf,
    const vector<float>& filter_vector,
    const owned_buffer::Buffer<bfloat16>& out_golden_tensor_buf,
    uint32_t reshard_num_cores_nhw = 0,
    bool remote_read = false) {
    log_debug(tt::LogTest, "Validating generate functions for config = {}", config);
    owned_buffer::Buffer<bfloat16> conv_tensor_buf;
    uint32_t diff;
    uint32_t failed_tests = 0;
    auto pad_metadata = generate_pad_metadata(config);
    auto tensor_metadata = generate_tensor_metadata(pad_metadata, config, reshard_num_cores_nhw);
    auto op_trace_metadata = generate_op_trace_metadata(config);
    auto shard_boundaries = generate_shard_boundaries(config, op_trace_metadata);
    auto sharded_input_top_left_indices =
        generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, false, false);
    auto halo_kernel_config =
        generate_halo_kernel_config_tensors(tensor_metadata, shard_boundaries, false, false, remote_read, device);

    auto [filter_h, filter_w] = config.window_hw;
    auto [input_h, input_w] = config.input_hw;
    auto [stride_h, stride_w] = config.stride_hw;
    auto output_shape = config.get_output_shape();
    uint32_t output_n, output_h, output_w;
    std::tie(output_n, output_h, output_w) = std::forward_as_tuple(output_shape[0], output_shape[1], output_shape[2]);

    uint32_t padded_input_h = input_h + 2 * config.pad_hw.first;
    uint32_t padded_input_w = input_w + 2 * config.pad_hw.second;

    auto ref_pad_metadata = pad_metadata_from_tensor_metadata(tensor_metadata);
    if (ref_pad_metadata != pad_metadata) {
        for (auto i = 0; i < ref_pad_metadata.size(); i++) {
            if (ref_pad_metadata[i] != pad_metadata[i]) {
                log_info(tt::LogTest, "Error at i = {}, Calculated = {}", i, bool(ref_pad_metadata[i]));
            }
        }
        log_error(
            tt::LogTest,
            "Failed to validate generate_tensor_metadata, convolution calculated with op_trace_metadata differs at "
            "locations = {}",
            diff);
        failed_tests++;
    }

    conv_tensor_buf = conv_using_op_trace_metadata(
        input_padded_tensor_buf,
        filter_vector,
        op_trace_metadata,
        stride_h,
        stride_w,
        filter_h,
        filter_w,
        padded_input_w,
        out_golden_tensor_buf.size());
    diff = compare_conv_out_with_golden(out_golden_tensor_buf, conv_tensor_buf);
    if (diff) {
        log_error(
            tt::LogTest,
            "Failed to validate generate_tensor_metadata, convolution calculated with op_trace_metadata differs at "
            "locations = {}",
            diff);
        failed_tests++;
    }

    conv_tensor_buf = conv_using_shard_boundaries(
        input_padded_tensor_buf,
        filter_vector,
        shard_boundaries,
        stride_h,
        stride_w,
        padded_input_h,
        padded_input_w,
        filter_h,
        filter_w,
        output_h,
        output_w,
        out_golden_tensor_buf.size());
    diff = compare_conv_out_with_golden(out_golden_tensor_buf, conv_tensor_buf);
    if (diff) {
        log_error(
            tt::LogTest,
            "Failed to validate generate_shard_boundaries, convolution calculated with op_trace_metadata differs at "
            "locations = {}",
            diff);
        failed_tests++;
    }

    conv_tensor_buf = conv_using_sliding_window_op_config(
        input_padded_tensor_buf,
        filter_vector,
        op_trace_metadata,
        shard_boundaries,
        sharded_input_top_left_indices,
        input_h,
        input_w,
        stride_h,
        stride_w,
        padded_input_w,
        filter_h,
        filter_w,
        out_golden_tensor_buf.size());
    diff = compare_conv_out_with_golden(out_golden_tensor_buf, conv_tensor_buf);
    if (diff) {
        log_error(
            tt::LogTest,
            "Failed to validate generate_sliding_window_op_config, convolution calculated with op_trace_metadata "
            "differs at locations = {}",
            diff);
        failed_tests++;
    }
    failed_tests +=
        validate_generate_halo_kernel_config(device, shard_boundaries, halo_kernel_config, pad_metadata, remote_read);
    return failed_tests;
}

// Container for testcase configurations
struct testcase_config {
    uint32_t batch_size;
    uint32_t input_h, input_w;
    uint32_t filter_h, filter_w;
    uint32_t stride_h, stride_w;
    uint32_t pad_h, pad_w;
    uint32_t num_cores_nhw;
    uint32_t reshard_num_cores_nhw;
    bool remote_read;
};

// Test cases
vector<struct testcase_config> configs = {
    // unique convs in rn50
    {64, 56, 56, 1, 1, 1, 1, 0, 0, 64, 0, false},
    {64, 56, 56, 1, 1, 2, 2, 0, 0, 64, 0, false},
    {64, 56, 56, 3, 3, 1, 1, 1, 1, 64, 0, false},

    {128, 56, 56, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {128, 28, 28, 3, 3, 1, 1, 1, 1, 64, 0, false},

    {256, 28, 28, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {256, 14, 14, 3, 3, 1, 1, 1, 1, 64, 0, false},

    {512, 14, 14, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {512, 7, 7, 3, 3, 1, 1, 1, 1, 64, 0, false},

    {16, 115, 115, 4, 4, 1, 1, 0, 0, 64, 0, false},
    // rn50 layer1
    {8, 56, 56, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {16, 56, 56, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {20, 56, 56, 3, 3, 1, 1, 1, 1, 64, 0, false},

    // rn50 layer2
    {8, 56, 56, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {16, 56, 56, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {20, 56, 56, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {8, 28, 28, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {16, 28, 28, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {20, 28, 28, 3, 3, 1, 1, 1, 1, 64, 0, false},

    // rn50 layer3
    {8, 28, 28, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {16, 28, 28, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {20, 28, 28, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {8, 14, 14, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {16, 14, 14, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {20, 14, 14, 3, 3, 1, 1, 1, 1, 64, 0, false},

    // rn50 layer4
    {8, 14, 14, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {16, 14, 14, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {20, 14, 14, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {8, 7, 7, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {16, 7, 7, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {20, 7, 7, 3, 3, 1, 1, 1, 1, 64, 0, false},

    // small test
    {1, 8, 8, 3, 3, 1, 1, 1, 1, 2, 0, false},
    {1, 16, 16, 3, 3, 1, 1, 1, 1, 4, 0, false},
    {8, 7, 7, 3, 3, 1, 1, 1, 1, 2, 0, false},

    // rn40 1x1s2 shapes
    {20, 56, 56, 1, 1, 2, 2, 0, 0, 64, 0, false},
    {20, 28, 28, 1, 1, 2, 2, 0, 0, 64, 0, false},
    {20, 14, 14, 1, 1, 2, 2, 0, 0, 64, 0, false},

    {8, 56, 56, 3, 3, 2, 2, 1, 1, 64, 0, false},

    // sd convs with HxW=64x64 with batch size 1
    {1, 64, 64, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 64, 64, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {1, 32, 32, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 32, 32, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {1, 16, 16, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 16, 16, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {1, 8, 8, 3, 3, 1, 1, 1, 1, 64, 0, false},
    // sd convs with HxW=64x64 with batch size 2
    {2, 64, 64, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {2, 64, 64, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {2, 32, 32, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {2, 32, 32, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {2, 16, 16, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {2, 16, 16, 3, 3, 2, 2, 1, 1, 64, 0, false},
    {2, 8, 8, 3, 3, 1, 1, 1, 1, 64, 0, false},

    {2, 64, 64, 1, 1, 1, 1, 1, 1, 64, 0, false},

    // unique convs in unet
    {1, 1056, 160, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 528, 80, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 264, 40, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 132, 20, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 66, 10, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 132, 20, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 264, 40, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 528, 80, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 1056, 160, 3, 3, 1, 1, 1, 1, 64, 0, false},

    // misc tests
    {1, 17, 17, 3, 3, 1, 1, 1, 1, 64, 0, false},
    {1, 23, 23, 3, 3, 1, 1, 1, 1, 64, 0, false},

    {1, 115, 115, 4, 4, 1, 1, 0, 0, 64, 0, false},

    {20, 28, 28, 3, 3, 2, 2, 1, 1, 64, 0, false},

    {8, 14, 14, 3, 3, 1, 1, 1, 1, 64, 0, false},

    {1, 60, 80, 3, 3, 1, 1, 1, 1, 64, 0, false},

    // tests for resharding, remote read
    {2, 5, 5, 3, 3, 2, 2, 1, 1, 1, 4, true},
    {3, 528, 80, 7, 7, 4, 4, 1, 1, 2, 0, true},
    {2, 10, 10, 7, 7, 4, 4, 3, 3, 4, 5, true},
    {7, 64, 64, 13, 13, 2, 2, 6, 6, 5, 4, true},

};

int main() {
    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    log_info(tt::LogTest, "Tests for Sliding window metadata calcations starts");
    for (auto tc : configs) {
        SlidingWindowConfig config{
            .batch_size = tc.batch_size,
            .input_hw = {tc.input_h, tc.input_w},
            .window_hw = {tc.filter_h, tc.filter_w},
            .stride_hw = {tc.stride_h, tc.stride_w},
            .pad_hw = {tc.pad_h, tc.pad_w},
            .dilation_hw = {1, 1},
            .num_cores_nhw = tc.num_cores_nhw};
        ttnn::SimpleShape input_tensor_shape(
            {config.batch_size,
             config.input_hw.first + 2 * config.pad_hw.first,
             config.input_hw.second + 2 * config.pad_hw.second});
        auto output_tensor_shape = config.get_output_shape();
        ttnn::SimpleShape filter_tensor_shape({config.window_hw.first, config.window_hw.second});

        Tensor input_padded_tensor =
            ttnn::random::random(input_tensor_shape, DataType::BFLOAT16).to(Layout::ROW_MAJOR).cpu();
        Tensor filter_tensor =
            ttnn::random::random(filter_tensor_shape, DataType::BFLOAT16).to(Layout::ROW_MAJOR).cpu();
        auto input_padded_tensor_buf = owned_buffer::get_as<bfloat16>(input_padded_tensor);
        auto filter_tensor_buf = owned_buffer::get_as<bfloat16>(filter_tensor);

        vector<float> filter_vector = create_filter_vec(filter_tensor_buf, tc.filter_h, tc.filter_w);
        owned_buffer::Buffer<bfloat16> out_golden_tensor_buf = ref_conv_op(
            input_padded_tensor,
            input_tensor_shape,
            tc.stride_h,
            tc.stride_w,
            filter_vector,
            filter_tensor_shape,
            output_tensor_shape);

        auto failed_tests = validate_generate_functions(
            device,
            config,
            input_padded_tensor_buf,
            filter_vector,
            out_golden_tensor_buf,
            tc.reshard_num_cores_nhw,
            tc.remote_read);
        if (failed_tests) {
            log_error(
                tt::LogTest,
                "Tests({}) failed for config ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
                failed_tests,
                tc.batch_size,
                tc.input_h,
                tc.input_w,
                tc.filter_h,
                tc.filter_w,
                tc.stride_h,
                tc.stride_w,
                tc.pad_h,
                tc.pad_w,
                tc.num_cores_nhw,
                tc.reshard_num_cores_nhw,
                tc.remote_read);
            TT_THROW("Tests Falied");
        } else {
            log_info(tt::LogTest, "Tests Passed");
        }
    }
    log_info(tt::LogTest, "Tests for Sliding window metadata calcations ends");
    TT_FATAL(tt::tt_metal::CloseDevice(device), "Error");
    return 0;
}
