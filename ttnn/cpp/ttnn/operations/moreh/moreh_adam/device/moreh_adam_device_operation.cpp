// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adam_device_operation.hpp"

#include <cstdint>

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_adam {
void MorehAdamOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& params_in = tensor_args.param_in;
    auto& grad = tensor_args.grad;
    auto& exp_avg_in = tensor_args.exp_avg_in;
    auto& exp_avg_sq_in = tensor_args.exp_avg_sq_in;

    tt::operations::primary::check_tensor(params_in, "moreh_adam", "params_in");
    tt::operations::primary::check_tensor(grad, "moreh_adam", "grad");
    tt::operations::primary::check_tensor(exp_avg_in, "moreh_adam", "exp_avg_in");
    tt::operations::primary::check_tensor(exp_avg_sq_in, "moreh_adam", "exp_avg_sq_in");

    if (tensor_args.max_exp_avg_sq_in) {
        tt::operations::primary::check_tensor(*tensor_args.max_exp_avg_sq_in, "moreh_adam", "max_exp_avg_sq_in");
    }

    const auto& params_out = tensor_args.output_tensors.at(0);

    if (params_out.has_value()) {
        tt::operations::primary::check_tensor(params_out.value(), "moreh_adam", "params_out");
    }

    if (tensor_args.output_tensors.at(1).has_value()) {
        tt::operations::primary::check_tensor(tensor_args.output_tensors.at(1).value(), "moreh_adam", "exp_avg_out");
    }

    if (tensor_args.output_tensors.at(2).has_value()) {
        tt::operations::primary::check_tensor(tensor_args.output_tensors.at(2).value(), "moreh_adam", "exp_avg_sq_out");
    }

    if (tensor_args.output_tensors.at(3).has_value()) {
        tt::operations::primary::check_tensor(
            tensor_args.output_tensors.at(3).value(), "moreh_adam", "max_exp_avg_sq_out");
    }
}

MorehAdamOperation::program_factory_t MorehAdamOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // For now we litteraly don't care and return a single factory. Whatever
    return ProgramFactory{};
}

void MorehAdamOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehAdamOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehAdamOperation::shape_return_value_t MorehAdamOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor_shape = tensor_args.param_in.get_shape();

    return {
        input_tensor_shape,
        input_tensor_shape,
        input_tensor_shape,
        input_tensor_shape,
    };
};

MorehAdamOperation::tensor_return_value_t MorehAdamOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    auto dtype = tensor_args.param_in.get_dtype();
    Layout layout{Layout::TILE};
    auto device = tensor_args.param_in.device();

    std::vector<std::optional<Tensor>> ret;
    auto output_mem_config = operation_attributes.output_mem_config;

    auto idx = uint32_t{0};
    if (tensor_args.output_tensors.at(idx).has_value()) {
        ret.push_back(tensor_args.output_tensors.at(idx).value());
    } else {
        ret.push_back(create_device_tensor(output_shapes.at(idx).value(), dtype, layout, device, output_mem_config));
    }
    ++idx;

    if (tensor_args.output_tensors.at(idx).has_value()) {
        ret.push_back(tensor_args.output_tensors.at(idx).value());
    } else {
        ret.push_back(create_device_tensor(output_shapes.at(idx).value(), dtype, layout, device, output_mem_config));
    }
    ++idx;

    if (tensor_args.output_tensors.at(idx).has_value()) {
        ret.push_back(tensor_args.output_tensors.at(idx).value());
    } else {
        ret.push_back(create_device_tensor(output_shapes.at(idx).value(), dtype, layout, device, output_mem_config));
    }
    ++idx;

    if (tensor_args.output_tensors.at(idx).has_value()) {
        ret.push_back(tensor_args.output_tensors.at(idx).value());
    } else if (operation_attributes.amsgrad) {
        ret.push_back(create_device_tensor(output_shapes.at(idx).value(), dtype, layout, device, output_mem_config));
    }

    return std::move(ret);
}

std::tuple<MorehAdamOperation::operation_attributes_t, MorehAdamOperation::tensor_args_t> MorehAdamOperation::invoke(
    const Tensor& param_in,
    const Tensor& grad,
    const Tensor& exp_avg_in,
    const Tensor& exp_avg_sq_in,

    const std::optional<float> lr,
    const std::optional<float> beta1,
    const std::optional<float> beta2,
    const std::optional<float> eps,
    const std::optional<float> weight_decay,
    const std::optional<uint32_t> step,
    const std::optional<bool> amsgrad,

    const std::optional<const Tensor> max_exp_avg_sq_in,
    const std::optional<const Tensor> param_out,
    const std::optional<const Tensor> exp_avg_out,
    const std::optional<const Tensor> exp_avg_sq_out,
    const std::optional<const Tensor> max_exp_avg_sq_out,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            lr.value_or(0.001f),
            beta1.value_or(0.9f),
            beta2.value_or(0.999f),
            eps.value_or(1e-8f),
            weight_decay.value_or(0.0f),
            step.value_or(0),
            amsgrad.value_or(false),
            memory_config.value_or(param_in.memory_config()),
            compute_kernel_config},
        tensor_args_t{
            param_in,
            grad,
            exp_avg_in,
            exp_avg_sq_in,
            max_exp_avg_sq_in,
            {param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out}}};
}
}  // namespace ttnn::operations::moreh::moreh_adam
