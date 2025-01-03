// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"

#include "tt_metal/host_api.hpp"

namespace ttnn::operations::experimental::transformer {

// Generic NLP CreateHeads op for decode
void NLPCreateHeadsDecodeDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_shape();
    // TODO: Rewrite validation for this decode case
    // NOTE: Checks for head_dim and shape[3] is done in nlp_create_qkv_heads because it's needed to infer head_dim
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16,
        "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Only tile layout is supported for input tensor");

    // input
    const uint32_t num_users_supported = 32;
    uint32_t num_users = input_shape[2];
    TT_FATAL(
        input_shape[3] % TILE_WIDTH == 0,
        "Unsupported input shape = {}",
        input_shape);  // head_dim must be multiple of TILE_WIDTH
    TT_FATAL(num_users <= num_users_supported, "Unsupported input shape = {}", input_shape);  // 32 users
    TT_FATAL(input_shape[1] == 1, "Unsupported input shape = {}", input_shape);
    TT_FATAL(input_shape[0] == 1, "Unsupported input shape = {}", input_shape);
    const auto QKV_memcfg = input_tensor.memory_config();
    if (input_tensor.is_sharded()) {
        TT_FATAL(
            QKV_memcfg.memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
            "Current input memory layout is {}. It must be width sharded",
            QKV_memcfg.memory_layout);
        TT_FATAL(
            input_tensor.shard_spec().value().shape[0] == input_tensor.volume() / input_tensor.get_legacy_shape()[-1],
            "Shard shape must be correct");
        TT_FATAL(
            input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
            "Shard orientation must be ROW_MAJOR");

        if (!this->overlap_qk_coregrid) {
            // Validate if each shard is a multiple of head_dim and doesn't contain partial heads
            TT_FATAL(
                this->head_dim % input_tensor.shard_spec().value().shape[1] == 0,
                "We don't support partial heads in shards when q and k heads are not overlapping coregrid");
        }
        TT_FATAL(
            !(this->batch_offset.has_value() ^ this->slice_size.has_value()),
            "Both batch_offset and slice_size must be provided or neither");
        if (this->batch_offset.has_value() && this->slice_size.has_value()) {
            TT_FATAL(this->batch_offset.value().get_shape()[0] == 1, "batch_offset must be unary tensor");
            uint32_t device_batch_offset = static_cast<uint32_t>(this->batch_offset.value().to_vector<int32_t>()[0]);
            TT_FATAL(
                device_batch_offset + this->slice_size.value() <= num_users,
                "Batch offset + slice size should be less than or equal to num_users");
        }

    } else {
        TT_FATAL(this->overlap_qk_coregrid, "Overlap_qk_coregrid must be true for non-sharded input");
    }

    // output
    TT_FATAL(
        this->output_mem_config.is_sharded() &&
            this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Output tensor must be height sharded");

    auto core_grid = input_tensor.device()->compute_with_storage_grid_size();

    // Support maximum 32 heads for now
    TT_FATAL(this->num_q_heads <= 32, "There are {} q heads only 32 are supported", this->num_q_heads);
    TT_FATAL(
        this->num_q_heads >= this->num_kv_heads,
        "num_q_heads={} must be greater than or equal to num_kv_heads={}",
        this->num_q_heads,
        this->num_kv_heads);

    uint32_t num_cores = core_grid.x * core_grid.y;
    // 1 User Per Core Max and 32 users for now
    if (this->overlap_qk_coregrid) {
        TT_FATAL(num_cores >= num_users, "Grid Size is {}. Need at least 32 cores for decode", num_cores);
    } else {
        TT_FATAL(
            num_cores >= 2 * num_users,
            "Grid Size is {}. Need cores atleast double of num_users for decode when q and k heads are not overlapping "
            "coregrid",
            num_cores);
    }
}

std::vector<tt::tt_metal::LegacyShape> NLPCreateHeadsDecodeDeviceOperation::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    std::vector<tt::tt_metal::LegacyShape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    auto batch = input_tensor.get_shape()[2];
    if (this->batch_offset.has_value() && this->slice_size.has_value()) {
        batch = this->slice_size.value();
    }
    auto head_dim = this->head_dim;

    // pad up to nearest multiple of TILE_HEIGHT for num_q_heads and num_kv_heads
    auto num_q_heads_padded = ((this->num_q_heads - 1) / TILE_HEIGHT + 1) * TILE_HEIGHT;
    auto num_kv_heads_padded = ((this->num_kv_heads - 1) / TILE_HEIGHT + 1) * TILE_HEIGHT;

    const tt::tt_metal::LegacyShape q_output_shape = tt::tt_metal::LegacyShape(
        {input_shape[0], batch, this->num_q_heads, head_dim}, {input_shape[0], batch, num_q_heads_padded, head_dim});
    const tt::tt_metal::LegacyShape v_output_shape = tt::tt_metal::LegacyShape(
        {input_shape[0], batch, this->num_kv_heads, head_dim}, {input_shape[0], batch, num_kv_heads_padded, head_dim});
    const tt::tt_metal::LegacyShape k_output_shape = v_output_shape;
    return {q_output_shape, k_output_shape, v_output_shape};
}

std::vector<Tensor> NLPCreateHeadsDecodeDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    auto output_shapes = this->compute_output_shapes(input_tensors);
    const auto& q_output_shape = output_shapes[0];

    auto batch = q_output_shape[1];
    auto num_q_heads_padded = ((this->num_q_heads - 1) / TILE_HEIGHT + 1) * TILE_HEIGHT;
    auto num_kv_heads_padded = ((this->num_q_heads - 1) / TILE_HEIGHT + 1) * TILE_HEIGHT;

    MemoryConfig q_mem_config = this->output_mem_config;
    MemoryConfig k_mem_config = this->output_mem_config;
    MemoryConfig v_mem_config = this->output_mem_config;
    CoreRangeSet q_shard_grid, k_shard_grid, v_shard_grid;
    if (!this->input_on_subcoregrids) {
        auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
        q_shard_grid = num_cores_to_corerangeset(batch, core_grid, true);
        if (this->overlap_qk_coregrid) {
            k_shard_grid = q_shard_grid;
        } else {
            k_shard_grid =
                num_cores_to_corerangeset(CoreCoord{batch % core_grid.x, batch / core_grid.x}, batch, core_grid, true);
        }
        v_shard_grid = q_shard_grid;

    } else {
        auto input_core_grid = input_tensor.shard_spec().value().grid;
        auto start_core_coord = input_core_grid.bounding_box().start_coord;
        q_shard_grid = num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch, input_core_grid, true);
        if (this->overlap_qk_coregrid) {
            k_shard_grid = q_shard_grid;
        } else {
            CoreRangeSet q_plus_one_grid =
                num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch + 1, input_core_grid, true);
            if (!q_plus_one_grid.ranges().empty()) {
                start_core_coord = q_plus_one_grid.ranges().back().end_coord;
            }
            k_shard_grid = num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch, input_core_grid, true);
        }
        v_shard_grid = q_shard_grid;
    }
    ShardSpec q_shard_spec{q_shard_grid, {num_q_heads_padded, this->head_dim}};
    q_mem_config.shard_spec = q_shard_spec;
    ShardSpec k_shard_spec{k_shard_grid, {num_kv_heads_padded, this->head_dim}};
    k_mem_config.shard_spec = k_shard_spec;
    ShardSpec v_shard_spec{v_shard_grid, {num_kv_heads_padded, this->head_dim}};
    v_mem_config.shard_spec = v_shard_spec;

    return {
        create_device_tensor(output_shapes[0], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), q_mem_config),
        create_device_tensor(output_shapes[1], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), k_mem_config),
        create_device_tensor(output_shapes[2], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), v_mem_config)
    };
}

operation::ProgramWithCallbacks NLPCreateHeadsDecodeDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    return multi_core_nlp_create_qkv_heads_decode(
        input_tensor,
        this->num_q_heads,
        this->num_kv_heads,
        this->head_dim,
        this->overlap_qk_coregrid,
        this->input_on_subcoregrids,
        this->batch_offset,
        this->slice_size,
        output_tensors,
        compute_with_storage_grid_size);
}

}  // namespace ttnn::operations::experimental::transformer
