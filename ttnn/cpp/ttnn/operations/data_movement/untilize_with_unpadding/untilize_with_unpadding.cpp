// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding.hpp"

#include "device/untilize_with_unpadding_op.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

using namespace tt::tt_metal;

ttnn::Shape squeeze_vector_shape(ttnn::Shape output_shape) {
    if (output_shape.rank() > 4) {
        ttnn::SmallVector<uint32_t> output_shape_4d(output_shape.rank());
        output_shape_4d[0] = 1;
        int extra_rank = output_shape.size() - 4;
        for (int i = extra_rank; i >= 0; i--) {
            output_shape_4d[0] *= (output_shape[i] + 1);
        }
        output_shape_4d[0]--;
        output_shape_4d[1] = output_shape[1 + extra_rank];
        output_shape_4d[2] = output_shape[2 + extra_rank];
        output_shape_4d[3] = output_shape[3 + extra_rank];
        return ttnn::Shape(std::move(output_shape_4d));
    }
    return output_shape;
}

namespace ttnn::operations::data_movement {

inline uint32_t get_estimated_size_of_cbs(
    const Tensor& input_tensor_a,
    const uint32_t input_single_tile_size,
    const uint32_t output_single_tile_size,
    const uint32_t num_tiles_per_row) {
    uint32_t cb_src0_size = input_single_tile_size * num_tiles_per_row;
    uint32_t cb_output_size = output_single_tile_size * num_tiles_per_row;
    return cb_src0_size + cb_output_size;
}

inline uint32_t get_max_l1_space(const Tensor& input_tensor_a) {
    auto device = input_tensor_a.device();
    auto lowest_address = device->lowest_occupied_compute_l1_address();
    uint32_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
    max_l1_space = max_l1_space - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    return max_l1_space;
}

inline bool enough_available_space(
    const Tensor& input_tensor_a,
    const uint32_t input_single_tile_size,
    const uint32_t output_single_tile_size,
    const uint32_t num_tiles_per_row) {
    uint32_t max_l1_space = get_max_l1_space(input_tensor_a);
    uint32_t estimated_size_of_cbs =
        get_estimated_size_of_cbs(input_tensor_a, input_single_tile_size, output_single_tile_size, num_tiles_per_row);
    return max_l1_space > estimated_size_of_cbs;
}

using OwnedUntilizeValArgs = std::tuple<ttnn::Tensor>;
using BaseUntilizeValType = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

using MassagedUntilizeVal = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&>;
using MassagedUntilizeValParams = MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&>;

MassagedUntilizeVal build_ndiml_untilize_val(BaseUntilizeValType base_untilize) {
    auto original_shape = std::make_shared<Shape>();

    return MassagedUntilizeVal(MassagedUntilizeValParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool {
            return input_tensor.get_logical_shape().rank() > 4;
        },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedUntilizeValArgs {
            *original_shape = input_tensor.get_logical_shape();
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto unsqueezed_tensor = ttnn::reshape(output, *original_shape);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_untilize)});
}

ttnn::Tensor ExecuteUntilizeWithUnpadding::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& output_tensor_end,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    bool use_pack_untilize) {
    // MT: Currently only uint32 is moved to DST directly, fp32 is converted to fp16b
    bool fp32_dest_acc_en = input_tensor.get_dtype() == DataType::UINT32;

    ttnn::SmallVector<uint32_t> output_end_vector;
    ttnn::Shape output_end;
    const auto input_shape = input_tensor.get_logical_shape();
    if (input_shape.rank() > 4) {
        for (auto index = 0; index < input_shape.rank(); ++index) {
            output_end_vector.push_back(input_shape[index] - 1);
        }
        output_end = squeeze_vector_shape(ttnn::Shape(std::move(output_end_vector)));
    } else {
        for (auto index = 0; index < input_tensor.get_logical_shape().rank(); ++index) {
            output_end_vector.push_back(output_tensor_end[index]);
        }
        output_end = ttnn::Shape(std::move(output_end_vector));
    }

    auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);

    uint32_t num_tiles_per_row = input_tensor.get_padded_shape()[-1] / tt::constants::TILE_WIDTH;
    uint32_t num_tiles_per_col = input_tensor.get_padded_shape()[-2] / tt::constants::TILE_HEIGHT;

    uint32_t output_single_tile_size = input_single_tile_size;
    bool enough_space_width =
        enough_available_space(input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_col);
    bool enough_space_height =
        enough_available_space(input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_row);

    auto base_untilize = [=](const ttnn::Tensor& input_tensor) {
        return operation::run(
            UntilizeWithUnpadding{// output_end,
                                  ttnn::Shape(output_end),
                                  memory_config.value_or(input_tensor.memory_config()),
                                  use_multicore,
                                  use_pack_untilize,
                                  fp32_dest_acc_en,
                                  enough_space_width,
                                  enough_space_height},
            {input_tensor},
            {},
            {},
            queue_id)[0];
    };

    return build_ndiml_untilize_val(base_untilize)(input_tensor);
}

}  // namespace ttnn::operations::data_movement
