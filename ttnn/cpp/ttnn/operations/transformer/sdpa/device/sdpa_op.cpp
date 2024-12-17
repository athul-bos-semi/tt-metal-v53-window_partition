// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_op.hpp"

#include "sdpa_program_factory.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::transformer {

void ScaledDotProductAttention::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    // Common validations for both modes
    TT_FATAL(input_tensors.size() == 3, "Must have 3 input tensors (Q, K, V)");
    TT_FATAL(
        optional_input_tensors.size() == 1 or optional_input_tensors.size() == 2,
        "Must have 1 or 2 optional tensors (mask/page_table)");

    for (auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to SDPA need to be on device");
        TT_FATAL(input_tensor.buffer() != nullptr, "Operands to SDPA need to be allocated in buffers on device");
        TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to SDPA must be tilized");
        TT_FATAL(
            input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B, "Error");
        TT_FATAL(
            input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to SDPA need to be in DRAM");
    }

    auto validate_regular_mode = [&]() {
        TT_FATAL(
            !(this->is_causal && optional_input_tensors.at(0).has_value()),
            "is_causal and attn_mask cannot both be present. Got is_causal: {}, attn_mask: {}",
            this->is_causal,
            optional_input_tensors.at(0).has_value());

        const auto& mask_option = optional_input_tensors.at(0);
        if (mask_option.has_value()) {
            auto mask = mask_option.value();
            TT_FATAL(
                mask.storage_type() == StorageType::DEVICE,
                "When mask is provided to SDPA, the tensor must be on device");
            TT_FATAL(
                input_tensors.at(0).device() == mask.device(),
                "When mask is provided to SDPA, it must be on the same device as the input tensors");
            TT_FATAL(mask.get_layout() == Layout::TILE, "When mask is provided to SDPA, it must be tilized");
            TT_FATAL(
                mask.get_dtype() == DataType::BFLOAT16 || mask.get_dtype() == DataType::BFLOAT8_B ||
                    mask.get_dtype() == DataType::BFLOAT4_B,
                "When mask is provided to SDPA, it must be in BF16, BFP8, or BFP4 dataformat");

            TT_FATAL(
                mask.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
                "When mask is provided to SDPA, it must be in DRAM");

            const auto mask_shape = mask.get_legacy_shape();
            const auto q_shape = input_tensors.at(0).get_legacy_shape();
            const auto k_shape = input_tensors.at(1).get_legacy_shape();

            TT_FATAL(mask_shape[0] == q_shape[0], "Mask batch dim must match Q batch dim");
            TT_FATAL(mask_shape[1] == 1, "Mask num_heads must be 1 to be broadcasted across all heads");
            TT_FATAL(mask_shape[2] == q_shape[2], "Mask sequence length must match Q sequence length");
            TT_FATAL(mask_shape[3] == k_shape[2], "Mask sequence length must match K sequence length");
        }

        // Shape checks
        const auto q_shape = input_tensors.at(0).get_legacy_shape();
        const auto k_shape = input_tensors.at(1).get_legacy_shape();
        const auto v_shape = input_tensors.at(2).get_legacy_shape();
        const auto B = q_shape[0];
        const auto nqh = q_shape[1];
        const auto nkv = k_shape[1];
        const auto Sq = q_shape[2];
        const auto DH = q_shape[3];
        const auto Sk = k_shape[2];
        if (this->is_causal) {
            TT_FATAL(
                Sq == Sk, "Causal SDPA requires Q and K to have the same sequence length. Got Q: {}, K: {}", Sq, Sk);
        }

        TT_FATAL(
            k_shape[0] == B && v_shape[0] == B, "K and V batch must match. Got K: {}, V: {}", k_shape[0], v_shape[0]);
        TT_FATAL(v_shape[1] == nkv, "K and V num_heads must match. Got K: {}, V: {}", k_shape[1], v_shape[1]);
        TT_FATAL(v_shape[2] == Sk, "K and V sequence length must match. Got K: {}, V: {}", k_shape[2], v_shape[2]);
        TT_FATAL(
            k_shape[3] == DH && v_shape[3] == DH,
            "K and V hidden dim must match. Got K: {}, V: {}",
            k_shape[3],
            v_shape[3]);
        TT_FATAL(
            nqh >= nkv && nqh % nkv == 0,
            "Q num_heads must be >= K num_heads and divisible by K num_heads. Got Q: {}, K: {}",
            nqh,
            nkv);

        if (this->program_config.has_value()) {
            auto q_chunk_size = program_config->q_chunk_size;
            auto k_chunk_size = program_config->k_chunk_size;

            TT_FATAL(
                Sq % q_chunk_size == 0,
                "q_chunk_size must divide q_shape[-2]. Got q_chunk_size: {}, q_shape[-2]: {}",
                q_chunk_size,
                q_shape[-2]);
            TT_FATAL(
                Sk % k_chunk_size == 0,
                "k_chunk_size must divide k_shape[-2]. Got k_chunk_size: {}, k_shape[-2]: {}",
                k_chunk_size,
                k_shape[-2]);
        }
    };

    auto validate_chunked_mode = [&]() {
        TT_FATAL(chunk_start_idx.has_value(), "chunk_start_idx must be provided for chunked mode");
        TT_FATAL(chunk_start_idx.value() >= 0, "chunk_start_idx must be non-negative");

        // Validate page table tensor
        const auto& page_table = optional_input_tensors[1].value();
        TT_FATAL(page_table.storage_type() == StorageType::DEVICE, "Page table tensor must be on device");
        TT_FATAL(
            input_tensors.at(0).device() == page_table.device(),
            "Page table must be on the same device as the input tensors");
        TT_FATAL(page_table.get_layout() == Layout::ROW_MAJOR, "Page table must be row major");
        // Check that page table is int32
        TT_FATAL(page_table.get_dtype() == DataType::INT32, "Page table must be int32");
        // Validate that first optional tensor (mask) is not provided
        TT_FATAL(
            !optional_input_tensors[0].has_value(),
            "Attention mask should not be provided in chunked mode - masking is handled internally");

        // Additional chunked-specific validations
        const auto q_shape = input_tensors.at(0).get_legacy_shape();
        const auto k_shape = input_tensors.at(1).get_legacy_shape();
        const auto v_shape = input_tensors.at(2).get_legacy_shape();
        const auto page_table_shape = page_table.get_legacy_shape();
        const auto B = q_shape[0];
        const auto nqh = q_shape[1];
        const auto nkv = k_shape[1];
        const auto Sq = q_shape[2];
        const auto DH = q_shape[3];
        const auto k_page_size = k_shape[2];
        const uint32_t num_pages_per_user = page_table.get_legacy_shape()[1];
        // Check that k page size matches v page size
        TT_FATAL(
            k_page_size == v_shape[2], "K page size must match V page size. Got K: {}, V: {}", k_page_size, v_shape[2]);
        // Check that page table has same batch size as input tensors
        TT_FATAL(
            page_table_shape[0] == B,
            "Page table batch size must match input batch size. Got Page table: {}, Input: {}",
            page_table_shape[0],
            B);
        // Calculate K length based on number of pages per user
        const uint32_t kv_length = num_pages_per_user * k_page_size;

        TT_FATAL(v_shape[1] == nkv, "K and V num_heads must match. Got K: {}, V: {}", k_shape[1], v_shape[1]);
        TT_FATAL(
            k_shape[3] == DH && v_shape[3] == DH,
            "K and V hidden dim must match. Got K: {}, V: {}",
            k_shape[3],
            v_shape[3]);
        TT_FATAL(
            nqh >= nkv && nqh % nkv == 0,
            "Q num_heads must be >= K num_heads and divisible by K num_heads. Got Q: {}, K: {}",
            nqh,
            nkv);

        if (this->program_config.has_value()) {
            auto q_chunk_size = program_config->q_chunk_size;
            auto k_chunk_size = program_config->k_chunk_size;

            TT_FATAL(
                Sq % q_chunk_size == 0,
                "q_chunk_size must divide q_shape[-2]. Got q_chunk_size: {}, q_shape[-2]: {}",
                q_chunk_size,
                q_shape[-2]);
            TT_FATAL(
                kv_length % k_chunk_size == 0,
                "k_chunk_size must divide k_shape[-2]. Got k_chunk_size: {}, k_shape[-2]: {}",
                k_chunk_size,
                k_shape[-2]);
        }

        // In chunked mode, K's sequence dimension should be >= Q's sequence dimension + chunk_start_idx
        TT_FATAL(
            kv_length >= q_shape[2] + chunk_start_idx.value(),
            "K's sequence length must be >= Q's sequence length + chunk_start_idx. Got K: {}, Q: {}, chunk_start_idx: "
            "{}",
            kv_length,
            q_shape[2],
            chunk_start_idx.value());
    };

    auto check_conditions = [&]() {
        bool has_chunk_start = chunk_start_idx.has_value();
        bool has_two_optional_inputs = optional_input_tensors.size() == 2;
        bool has_page_table = optional_input_tensors.size() > 1 && optional_input_tensors.at(1).has_value();
        TT_FATAL(
            has_chunk_start == has_two_optional_inputs, "chunk_start_idx and number of optional inputs must match");
        TT_FATAL(
            has_two_optional_inputs == has_page_table,
            "page_table must be provided if and only if there are two optional inputs");
    };

    check_conditions();
    bool is_chunked_mode = chunk_start_idx.has_value();

    // Check if we're in chunked mode and call appropriate validation
    if (is_chunked_mode) {
        validate_chunked_mode();
    } else {
        validate_regular_mode();
    }
}

std::vector<TensorSpec> ScaledDotProductAttention::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    return {TensorSpec(
        input.get_logical_shape(), TensorLayout(input.get_dtype(), PageConfig(Layout::TILE), output_mem_config))};
}

operation::ProgramWithCallbacks ScaledDotProductAttention::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    const auto& attn_mask = optional_input_tensors.at(0);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.get_legacy_shape()[-1]));
    }

    std::size_t q_chunk_size = this->program_config ? this->program_config->q_chunk_size : 32;
    std::size_t k_chunk_size = this->program_config ? this->program_config->k_chunk_size : 32;
    // get page table if chunked
    const auto page_table = this->chunk_start_idx.has_value() ? optional_input_tensors.at(1) : std::nullopt;

    return detail::sdpa_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        output_tensor,
        attn_mask,
        page_table,
        this->chunk_start_idx,
        scale,
        this->is_causal,
        q_chunk_size,
        k_chunk_size,
        this->compute_kernel_config,
        this->program_config);
}

operation::Hash ScaledDotProductAttention::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    bool is_chunked_prefill = this->chunk_start_idx.has_value();
    return operation::hash_operation<ScaledDotProductAttention>(
        this->scale,
        this->output_mem_config,
        this->program_config,
        this->is_causal,
        is_chunked_prefill,
        this->compute_kernel_config,
        input_tensors,
        optional_input_tensors);
}

}  // namespace ttnn::operations::transformer
