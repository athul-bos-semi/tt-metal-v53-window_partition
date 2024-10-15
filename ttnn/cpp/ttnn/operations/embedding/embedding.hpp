// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

namespace operations {

namespace embedding {

struct EmbeddingOperation {
    static inline Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor_arg,
        const Tensor& weight_arg,
        const std::optional<int>& pad_token = std::nullopt,
        const Layout& layout = ttnn::ROW_MAJOR_LAYOUT,
        EmbeddingsType embeddings_type = EmbeddingsType::GENERIC,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        if (pad_token.has_value()) {
            embeddings_type = EmbeddingsType::PADDED;
        }
        Tensor mutable_input_tensor = input_tensor_arg;
        Tensor mutable_weight = weight_arg;
        if (mutable_input_tensor.get_layout() == ttnn::TILE_LAYOUT) {
            mutable_input_tensor = ttnn::to_layout(mutable_input_tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
        }
        if (mutable_weight.get_layout() == ttnn::TILE_LAYOUT) {
            mutable_weight = ttnn::to_layout(mutable_weight, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
        }
        auto hidden_embedding_dim = mutable_weight.get_shape()[-1];
        auto padded_hidden_embedding_dim = mutable_weight.get_shape().with_tile_padding()[-1];
        auto weight = ttnn::unsqueeze_to_4D(mutable_weight);

        auto batch_size = mutable_input_tensor.get_shape()[0];
        auto sentence_size = mutable_input_tensor.get_shape()[-1];
        auto input_tensor =
            ttnn::reshape(mutable_input_tensor, ttnn::Shape{std::array<uint32_t, 4>{batch_size, 1, 1, sentence_size}});

        bool fuzed_tilized = layout == ttnn::TILE_LAYOUT;

        // If layout is row major, OR if the input tensor is not a multiple of TILE_HEIGHT, then we cannot use tilized
        if(!fuzed_tilized || input_tensor.get_legacy_shape()[-1] % TILE_HEIGHT) fuzed_tilized = false;
        if(!fuzed_tilized || weight.get_legacy_shape()[-1] % TILE_WIDTH) fuzed_tilized = false;

        auto embeddings = operation::run(
                                Embeddings{
                                    .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                                    .tilized = fuzed_tilized,
                                    .embeddings_type = embeddings_type,
                                    .pad_token = pad_token,
                                    .output_dtype = dtype.value_or(weight.get_dtype())},
                                {input_tensor, weight})
                                .at(0);
        embeddings = ttnn::reshape(
            embeddings, ttnn::Shape{std::array<uint32_t, 3>{batch_size, sentence_size, hidden_embedding_dim}});
        embeddings = ttnn::to_layout(embeddings, layout, std::nullopt, std::nullopt, (Device*)nullptr);
        return embeddings;
    }
    static inline auto invoke(
        const Tensor& input_tensor_arg,
        const Tensor& weight_arg,
        const std::optional<int>& pad_token = std::nullopt,
        const Layout& layout = ttnn::ROW_MAJOR_LAYOUT,
        EmbeddingsType embeddings_type = EmbeddingsType::GENERIC,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt
        ) {
            return invoke(DefaultQueueId, input_tensor_arg, weight_arg, pad_token, layout, embeddings_type, dtype, memory_config, optional_output_tensor);
        }
};

}  // namespace embedding
}  // namespace operations

constexpr auto embedding = ttnn::register_operation_with_auto_launch_op<"ttnn::embedding", ttnn::operations::embedding::EmbeddingOperation>();

}  // namespace ttnn
