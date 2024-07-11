// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <reflect>
#include <tuple>
#include <type_traits>

#include "tensor/tensor.hpp"
#include "third_party/json/json.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tools/profiler/profiler.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/third_party/tracy/public/tracy/TracyC.h"

using json = nlohmann::json;

namespace tt {

namespace tt_metal {

extern std::atomic<uint32_t> operation_id_atomic_count;

inline uint32_t assign_operation_id() {
    return operation_id_atomic_count.fetch_add(1);
}

namespace op_profiler {

enum class OpType { python_fallback, tt_dnn_cpu, tt_dnn_device, unknown };

#if defined(TRACY_ENABLE)
class thread_safe_cached_ops_map {
    using OP_INFO_MAP = std::unordered_map<tt::tt_metal::operation::Hash, std::string>;
    using DEVICE_OP_MAP = std::unordered_map<uint32_t, OP_INFO_MAP>;

   public:
    DEVICE_OP_MAP::iterator find(uint32_t device_id) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map.find(device_id);
    }
    DEVICE_OP_MAP::iterator end() {
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map.end();
    }
    OP_INFO_MAP& at(uint32_t device_id) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        return map.at(device_id);
    }
    void emplace(uint32_t device_id, OP_INFO_MAP&& device_op_entry) {
        std::scoped_lock<std::mutex> lock(map_mutex);
        map.emplace(device_id, device_op_entry);
    }

   private:
    std::mutex map_mutex;
    DEVICE_OP_MAP map;
};

class thread_safe_call_stack {
   public:
    void push(const TracyCZoneCtx& ctx) {
        std::scoped_lock<std::mutex> lock(stack_mutex);
        call_stack.push(ctx);
    }
    bool empty() {
        std::scoped_lock<std::mutex> lock(stack_mutex);
        return call_stack.empty();
    }
    void pop() {
        std::scoped_lock<std::mutex> lock(stack_mutex);
        call_stack.pop();
    }
    TracyCZoneCtx& top() {
        std::scoped_lock<std::mutex> lock(stack_mutex);
        return call_stack.top();
    }

   private:
    std::mutex stack_mutex;
    stack<TracyCZoneCtx> call_stack;
};

inline thread_safe_cached_ops_map cached_ops{};
inline thread_safe_call_stack call_stack;
#endif

static void start_tracy_zone(const string& source, const string& functName, uint32_t lineNum, uint32_t color = 0) {
#if defined(TRACY_ENABLE)
    auto tracySrcLoc =
        ___tracy_alloc_srcloc(lineNum, source.c_str(), source.length(), functName.c_str(), functName.length());
    TracyCZoneCtx ctx = ___tracy_emit_zone_begin_alloc(tracySrcLoc, 1);
    if (color != 0) {
        TracyCZoneColor(ctx, color);
    }

    call_stack.push(ctx);
#endif
}

static bool stop_tracy_zone(const string& name = "", uint32_t color = 0) {
    bool callStackWasEmpty = true;
#if defined(TRACY_ENABLE)
    if (!call_stack.empty()) {
        callStackWasEmpty = false;
        TracyCZoneCtx ctx = call_stack.top();
        if (name != "") {
            TracyCZoneName(ctx, name.c_str(), name.length());
        }
        if (color != 0) {
            TracyCZoneColor(ctx, color);
        }
        TracyCZoneEnd(ctx);
        call_stack.pop();
    }
#endif
    return callStackWasEmpty;
}

static void tracy_message(const string& source, uint32_t color = 0xf0f8ff) {
    TracyMessageC(source.c_str(), source.size(), color);
}

static void tracy_frame() { FrameMark; }

#if defined(TRACY_ENABLE)
static inline json get_kernels_json(const Program& program) {
    vector<json> computeKernels;
    vector<json> datamovementKernels;
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        auto kernel = tt::tt_metal::detail::GetKernel(program, kernel_id).get();
        if (kernel->processor() == RISCV::COMPUTE) {
            ComputeKernel* computeKernel = static_cast<ComputeKernel*>(kernel);
            MathFidelity mathFidelity = std::get<ComputeConfig>(computeKernel->config()).math_fidelity;
            json computeKernelObj;
            computeKernelObj["math_fidelity"] = fmt::format("{}", magic_enum::enum_name(mathFidelity));
            computeKernelObj["path"] = computeKernel->kernel_path_file_name();
            computeKernelObj["name"] = computeKernel->get_full_kernel_name();
            computeKernels.push_back(computeKernelObj);
        } else {
            json datamovementKernelObj;
            datamovementKernelObj["path"] = kernel->kernel_path_file_name();
            datamovementKernelObj["name"] = kernel->get_full_kernel_name();
            datamovementKernels.push_back(datamovementKernelObj);
        }
    }
    json ret;
    ret["compute_kernels"] = computeKernels;
    ret["datamovement_kernels"] = datamovementKernels;
    return ret;
}

static inline json get_tensor_json(const Tensor& tensor) {
    json ret;
    string tensorStorageStr;
    if (tensor.storage_type() == StorageType::DEVICE) {
        ret["storage_type"]["device_id"] = tensor.device()->id();
        ret["storage_type"]["memory_config"]["buffer_type"] = magic_enum::enum_name(tensor.memory_config().buffer_type);
        ret["storage_type"]["memory_config"]["memory_layout"] =
            magic_enum::enum_name(tensor.memory_config().memory_layout);
    } else {
        ret["storage_type"] = fmt::format("{}", magic_enum::enum_name(tensor.storage_type()));
    }

    auto tensor_shape = tensor.get_legacy_shape();
    ret["shape"]["W"] = tensor_shape.rank() >= 4 ? tensor_shape[-4] : 1;
    ret["shape"]["Z"] = tensor_shape.rank() >= 3 ? tensor_shape[-3] : 1;
    ret["shape"]["Y"] = tensor_shape.rank() >= 2 ? tensor_shape[-2] : 1;
    ret["shape"]["X"] = tensor_shape[-1];
    ret["layout"] = fmt::format("{}", magic_enum::enum_name(tensor.get_layout()));
    ret["dtype"] = fmt::format("{}", magic_enum::enum_name(tensor.get_dtype()));

    return ret;
}

static inline vector<json> get_tensors_json(const vector<Tensor>& tensors) {
    ZoneScoped;
    vector<json> ret;
    for (auto& tensor : tensors) {
        ret.push_back(get_tensor_json(tensor));
    }
    return ret;
}

static inline vector<json> get_tensors_json(const vector<std::optional<const Tensor>>& tensors) {
    ZoneScoped;
    vector<json> ret;
    for (auto& tensor : tensors) {
        if (tensor.has_value()) {
            ret.push_back(get_tensor_json(tensor.value()));
        }
    }
    return ret;
}

static inline vector<json> get_tensors_json(const vector<std::optional<Tensor>>& tensors) {
    ZoneScoped;
    vector<json> ret;
    for (auto& tensor : tensors) {
        if (tensor.has_value()) {
            ret.push_back(get_tensor_json(tensor.value()));
        }
    }
    return ret;
}

template <bool IsExternal = false, typename Operation>
inline json get_base_json(
    uint32_t opID,
    const Operation& op,
    const std::vector<Tensor>& input_tensors,
    std::optional<std::reference_wrapper<typename Operation::OutputTensors>> output_tensors = std::nullopt) {
    ZoneScoped;
    json j;
    j["global_call_count"] = opID;

    std::string opName = op.get_type_name();

    if constexpr (!IsExternal) {
        auto profiler_info = op.create_profiler_info(input_tensors);
        if (profiler_info.preferred_name.has_value()) {
            j["op_code"] = profiler_info.preferred_name.value();
        }

        if (profiler_info.parallelization_strategy.has_value()) {
            j["parallelization_strategy"] = profiler_info.parallelization_strategy.value();
        }
    }

    std::replace(opName.begin(), opName.end(), ',', ';');
    j["op_code"] = opName;

    json attributesObj;
    auto attributes = op.attributes();
    if (not attributes.empty()) {
        ZoneScopedN("get_attributes_json");
        for (auto&& [name, value] : attributes) {
            std::string nameStr = "";
            nameStr = fmt::format("{}", name);
            attributesObj[nameStr] = fmt::format("{}", value);
        }
    }

    j["attributes"] = attributesObj;

    j["input_tensors"] = get_tensors_json(input_tensors);

    if (output_tensors.has_value()) {
        j["output_tensors"] = get_tensors_json(output_tensors.value());
    }
    return j;
}

template <typename operation_t>
inline json get_base_json(
    uint32_t operation_id,
    const typename operation_t::operation_attributes_t& operation_attributes,
    const typename operation_t::tensor_args_t& tensor_args,
    typename operation_t::tensor_return_value_t& tensor_return_value) {
    ZoneScoped;
    json j;
    j["global_call_count"] = operation_id;

    auto as_string = [](std::string_view v) -> std::string { return {v.data(), v.size()}; };
    std::string opName = as_string(tt::stl::get_type_name<operation_t>());
    std::replace(opName.begin(), opName.end(), ',', ';');
    j["op_code"] = opName;

    json attributesObj;
    reflect::for_each(
        [&attributesObj, &operation_attributes](auto I) {
            attributesObj[std::string{reflect::member_name<I>(operation_attributes)}] =
                fmt::format("{}", reflect::get<I>(operation_attributes));
        },
        operation_attributes);
    j["attributes"] = attributesObj;

    std::vector<json> input_tensors;
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&input_tensors](auto&& tensor) { input_tensors.push_back(get_tensor_json(tensor)); }, tensor_args);
    j["input_tensors"] = input_tensors;

    std::vector<json> output_tensors;
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&output_tensors](auto&& tensor) { output_tensors.push_back(get_tensor_json(tensor)); }, tensor_return_value);
    j["output_tensors"] = output_tensors;

    return j;
}

inline std::string op_meta_data_serialized_json(
    uint32_t opID, const tt::tt_metal::operation::ExternalOperation& op, const std::vector<Tensor>& input_tensors) {
    auto j = get_base_json<true>(opID, op, input_tensors);
    j["op_type"] = magic_enum::enum_name(OpType::python_fallback);
    std::string ser = j.dump(4);
    return fmt::format("`TT_DNN_FALL_BACK_OP:{} ->\n{}`", j["op_code"], ser);
}

template <typename OutputTensors, template <typename> typename HostOperationType>
inline std::string op_meta_data_serialized_json(
    uint32_t opID,
    const HostOperationType<OutputTensors>& op,
    const std::vector<Tensor>& input_tensors,
    OutputTensors& output_tensors) {
    auto j = get_base_json(opID, op, input_tensors, output_tensors);
    j["op_type"] = magic_enum::enum_name(OpType::tt_dnn_cpu);
    std::string ser = j.dump(4);
    return fmt::format("`TT_DNN_HOST_OP:{} ->\n{}`", j["op_code"], ser);
}

template <typename OutputTensors, template <typename> typename DeviceOperationType>
inline std::string op_meta_data_serialized_json(
    uint32_t opID,
    tt::tt_metal::operation::Hash opHash,
    bool isProgramCached,
    uint32_t device_id,
    const DeviceOperationType<OutputTensors>& op,
    const std::variant<std::shared_ptr<Program>, std::reference_wrapper<Program>>& program,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
    OutputTensors& output_tensors) {
    const bool useCachedOps = std::getenv("TT_METAL_PROFILER_NO_CACHE_OP_INFO") == nullptr;
    if (!useCachedOps || !isProgramCached || (cached_ops.find(device_id) == cached_ops.end()) ||
        (cached_ops.at(device_id).find(opHash) == cached_ops.at(device_id).end())) {
        auto j = get_base_json(opID, op, input_tensors, output_tensors);
        j["op_type"] = magic_enum::enum_name(OpType::tt_dnn_device);
        j["device_id"] = device_id;
        j["op_hash"] = opHash;
        if (std::holds_alternative<std::reference_wrapper<Program>>(program)) {
            j["kernel_info"] = get_kernels_json(std::get<std::reference_wrapper<Program>>(program));
        } else if (std::holds_alternative<std::shared_ptr<Program>>(program)) {
            auto prg = std::get<std::shared_ptr<Program>>(program);
            if (prg != nullptr) {
                j["kernel_info"] = get_kernels_json(*prg);
            }
        }

        j["optional_input_tensors"] = get_tensors_json(optional_input_tensors);

        auto perfModel = op.create_op_performance_model(input_tensors, optional_input_tensors, output_tensors);
        j["performance_model"]["compute_ns"] = perfModel.get_compute_ns();
        j["performance_model"]["ideal_ns"] = perfModel.get_ideal_ns();
        j["performance_model"]["bandwidth_ns"] = perfModel.get_bandwidth_ns();
        j["performance_model"]["input_bws"] = perfModel.get_input_bws();
        j["performance_model"]["output_bws"] = perfModel.get_output_bws();

        std::string short_str = fmt::format("`TT_DNN_DEVICE_OP: {}, {}, {}, ", j["op_code"], opHash, device_id);
        if (cached_ops.find(device_id) == cached_ops.end()) {
            cached_ops.emplace(
                device_id, (std::unordered_map<tt::tt_metal::operation::Hash, std::string>){{opHash, short_str}});
        } else {
            cached_ops.at(device_id).emplace(opHash, short_str);
        }

        std::string ser = j.dump(4);
        return fmt::format("{}{} ->\n{}`", short_str, opID, ser);
    } else {
        return fmt::format("{}{}`", cached_ops.at(device_id).at(opHash), opID);
    }
}

template <typename operation_t>
inline std::string op_meta_data_serialized_json(
    const operation_t& operation,
    uint32_t operation_id,
    auto device_id,
    const auto& program,
    const auto& program_hash,
    const auto& operation_attributes,
    const auto& tensor_args,
    auto& tensor_return_value) {
    const bool useCachedOps = std::getenv("TT_METAL_PROFILER_NO_CACHE_OP_INFO") == nullptr;
    if (!useCachedOps || (cached_ops.find(device_id) == cached_ops.end()) ||
        (cached_ops.at(device_id).find(program_hash) == cached_ops.at(device_id).end())) {
        auto j = get_base_json<operation_t>(operation_id, operation_attributes, tensor_args, tensor_return_value);
        j["op_type"] = magic_enum::enum_name(OpType::tt_dnn_device);
        j["device_id"] = device_id;
        j["op_hash"] = program_hash;
        j["kernel_info"] = get_kernels_json(program);

        j["optional_input_tensors"] = std::vector<json>{};

        auto perfModel = [&]() {
            if constexpr (requires { operation_t::create_op_performance_model; }) {
                return operation_t::create_op_performance_model(operation_attributes, tensor_args, tensor_return_value);
            } else {
                return operation::OpPerformanceModel{};
            }
        }();
        j["performance_model"]["compute_ns"] = perfModel.get_compute_ns();
        j["performance_model"]["ideal_ns"] = perfModel.get_ideal_ns();
        j["performance_model"]["bandwidth_ns"] = perfModel.get_bandwidth_ns();
        j["performance_model"]["input_bws"] = perfModel.get_input_bws();
        j["performance_model"]["output_bws"] = perfModel.get_output_bws();

        std::string short_str = fmt::format("`TT_DNN_DEVICE_OP: {}, {}, {}, ", j["op_code"], program_hash, device_id);
        if (cached_ops.find(device_id) == cached_ops.end()) {
            cached_ops.emplace(
                device_id, (std::unordered_map<tt::tt_metal::operation::Hash, std::string>){{program_hash, short_str}});
        } else {
            cached_ops.at(device_id).emplace(program_hash, short_str);
        }

        std::string ser = j.dump(4);
        return fmt::format("{}{} ->\n{}`", short_str, operation_id, ser);
    } else {
        return fmt::format("{}{}`", cached_ops.at(device_id).at(program_hash), operation_id);
    }
}

#define TracyOpTTNNDevice(                                                                                           \
    op_id, op_hash, is_cached, device_id, operation, program, input_tensors, optional_input_tensors, output_tensors) \
    std::string op_message = op_profiler::op_meta_data_serialized_json(                                              \
        op_id,                                                                                                       \
        op_hash,                                                                                                     \
        is_cached,                                                                                                   \
        device_id,                                                                                                   \
        operation,                                                                                                   \
        program,                                                                                                     \
        input_tensors,                                                                                               \
        optional_input_tensors,                                                                                      \
        output_tensors);                                                                                             \
    std::string op_text = fmt::format("id:{}", op_id);                                                               \
    ZoneText(op_text.c_str(), op_text.size());                                                                       \
    TracyMessage(op_message.c_str(), op_message.size());

#define TracyOpTNNNDeviceV2(                                                                                           \
    operation, operation_id, device_id, program, program_hash, operation_attributes, tensor_args, tensor_return_value) \
    std::string op_message = op_profiler::op_meta_data_serialized_json(                                                \
        operation,                                                                                                     \
        operation_id,                                                                                                  \
        device_id,                                                                                                     \
        program,                                                                                                       \
        program_hash,                                                                                                  \
        operation_attributes,                                                                                          \
        tensor_args,                                                                                                   \
        tensor_return_value);                                                                                          \
    std::string op_text = fmt::format("id:{}", operation_id);                                                          \
    ZoneText(op_text.c_str(), op_text.size());                                                                         \
    TracyMessage(op_message.c_str(), op_message.size());

#define TracyOpTTNNHost(op_id, operation, input_tensors, output_tensors)                            \
    std::string op_message =                                                                        \
        op_profiler::op_meta_data_serialized_json(op_id, operation, input_tensors, output_tensors); \
    std::string op_text = fmt::format("id:{}", op_id);                                              \
    ZoneText(op_text.c_str(), op_text.size());                                                      \
    TracyMessage(op_message.c_str(), op_message.size());

#define TracyOpTTNNExternal(op_id, op, input_tensors)                                             \
    std::string op_message = op_profiler::op_meta_data_serialized_json(op_id, op, input_tensors); \
    std::string op_text = fmt::format("id:{}", op_id);                                            \
    ZoneText(op_text.c_str(), op_text.size());                                                    \
    TracyMessage(op_message.c_str(), op_message.size());

#else

#define TracyOpTTNNDevice( \
    op_id, op_hash, is_cached, device_id, operation, program, input_tensors, optional_input_tensors, output_tensors)
#define TracyOpTNNNDeviceV2( \
    operation, operation_id, device_id, program, program_hash, operation_attributes, tensor_args, tensor_return_value)
#define TracyOpTTNNHost(op_id, operation, input_tensors, output_tensors)
#define TracyOpTTNNExternal(op_id, op, input_tensors)

#endif
}  // namespace op_profiler
}  // namespace tt_metal
}  // namespace tt
