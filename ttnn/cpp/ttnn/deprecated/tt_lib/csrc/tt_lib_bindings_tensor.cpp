// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "ttnn/cpp/pybind11/json_class.hpp"

#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/auto_format.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/layernorm_distributed/layernorm_pre_allgather_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/layernorm_distributed/layernorm_post_allgather_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/update_cache/update_cache_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_lib_bindings.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "type_caster.hpp"

namespace tt::tt_metal {

namespace detail {
template <class T>
struct DataTypeToFormatType {
    using type = T;
};

template <>
struct DataTypeToFormatType<bfloat16> {
    using type = uint16_t;
};

template <class CppType, class DataType, class PyType>
void implement_buffer_protocol(PyType& py_buffer_t) {
    py_buffer_t.def("__getitem__", [](const CppType& self, std::size_t index) { return self[index]; })
        .def("__len__", [](const CppType& self) { return self.size(); })
        .def(
            "__iter__",
            [](const CppType& self) { return py::make_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>())
        .def_buffer([](CppType& self) -> py::buffer_info {
            using FormatType = typename DataTypeToFormatType<DataType>::type;
            return py::buffer_info(
                self.begin(),                                /* Pointer to buffer */
                sizeof(DataType),                            /* Size of one scalar */
                py::format_descriptor<FormatType>::format(), /* Python struct-style format descriptor */
                1,                                           /* Number of dimensions */
                {self.size()},                               /* Buffer dimensions */
                {sizeof(DataType)}                           /* Strides (in bytes) for each index */
            );
        });
};

}  // namespace detail

void TensorModule(py::module& m_tensor) {
    // ENUM SECTION

    // layout enums
    detail::export_enum<Layout>(m_tensor);

    detail::export_enum<DataType>(m_tensor);

    detail::export_enum<StorageType>(m_tensor);

    detail::export_enum<MathFidelity>(m_tensor);

    detail::export_enum<TensorMemoryLayout>(m_tensor);

    detail::export_enum<ShardOrientation>(m_tensor);


    py::enum_<BufferType>(m_tensor, "BufferType")
        .value("DRAM", BufferType::DRAM)
        .value("L1", BufferType::L1)
        .value("L1_SMALL", BufferType::L1_SMALL);


    auto py_core_coord = tt_serializable_class<CoreCoord>(m_tensor, "CoreCoord", R"doc(
        Class defining core coordinate
    )doc");

    py_core_coord.def(py::init<std::size_t, std::size_t>())
        .def(py::init<>([](std::tuple<std::size_t, std::size_t> core_coord) {
            return CoreCoord(std::get<0>(core_coord), std::get<1>(core_coord));
        }))
        .def("__repr__", [](const CoreCoord& self) -> std::string { return self.str(); })
        .def_readonly("x", &CoreCoord::x)
        .def_readonly("y", &CoreCoord::y);
    py::implicitly_convertible<std::tuple<std::size_t, std::size_t>, CoreCoord>();

    auto py_shape = py::class_<Shape>(m_tensor, "Shape", R"doc(
        Class defining tensor shape
    )doc");

    py_shape.def(py::init<std::array<uint32_t, 4>>())
        .def(
            py::init(
                [](const std::vector<uint32_t>& shape,
                   const std::optional<std::vector<uint32_t>>& padded_shape) -> Shape {
                    if (padded_shape.has_value()) {
                        return Shape{shape, padded_shape.value()};
                    } else {
                        return Shape{shape};
                    }
                }),
            py::arg("shape"),
            py::arg("padded_shape") = std::nullopt)
        .def("__len__", [](const Shape& self) { return self.rank(); })
        .def("__eq__", [](const Shape& self, const Shape& other) { return self == other; })
        .def("__eq__", [](const Shape& self, const std::vector<uint32_t>& other) { return self == Shape{other}; })
        .def("__eq__", [](const Shape& self, const std::array<uint32_t, 4>& other) { return self == Shape{other}; })
        .def("__eq__", [](const Shape& self, const py::none) { return false; })
        .def("__getitem__", [](const Shape& self, const std::int64_t index) { return self[index]; })
        .def(
            "__getitem__",
            [](const Shape& self, const py::slice slice) {
                size_t start = 0, stop = 0, step = 0, slicelength = 0;
                if (!slice.compute(self.rank(), &start, &stop, &step, &slicelength)) {
                    throw std::runtime_error("Invalid slice");
                }

                std::vector<uint32_t> output;
                for (auto index = start; index < stop; index += step) {
                    output.push_back(self[index]);
                }
                return Shape{output};
            })
        .def(
            "__iter__",
            [](const Shape& self) { return py::make_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>())
        .def("__repr__", [](const Shape& self) { return fmt::format("{}", self); })
        .def("without_padding", [](const Shape& self) -> Shape { return self.without_padding(); });

    py::implicitly_convertible<std::vector<uint32_t>, Shape>();

    auto pyMemoryConfig = tt_serializable_class<MemoryConfig>(m_tensor, "MemoryConfig", R"doc(
        Class defining memory configuration for storing tensor data on TT Accelerator device.
        There are eight DRAM memory banks on TT Accelerator device, indexed as 0, 1, 2, ..., 7.
    )doc");

    pyMemoryConfig
        .def(
            py::init<>(
                [](TensorMemoryLayout memory_layout, BufferType buffer_type, std::optional<ShardSpec> shard_spec) {
                    return MemoryConfig{
                        .memory_layout = memory_layout, .buffer_type = buffer_type, .shard_spec = shard_spec};
                }),
            py::arg("memory_layout") = TensorMemoryLayout::INTERLEAVED,
            py::arg("buffer_type") = BufferType::DRAM,
            py::arg("shard_spec") = std::nullopt,
            R"doc(
                Create MemoryConfig class.
                If interleaved is set to True, tensor data will be interleaved across multiple DRAM banks on TT Accelerator device.
                Otherwise, tensor data will be stored in a DRAM bank selected by dram_channel (valid values are 0, 1, ..., 7).

                Example of creating MemoryConfig specifying that tensor data should be stored in DRAM bank 3.

                .. code-block:: python

                    mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.SINGLE_BANK)
            )doc")
        .def(
            "__hash__",
            [](const MemoryConfig& memory_config) -> tt::stl::hash::hash_t {
                return tt::stl::hash::detail::hash_object(memory_config);
            })
        .def("is_sharded", &MemoryConfig::is_sharded, "Whether tensor data is sharded across multiple cores in L1")
        .def_property_readonly(
            "interleaved",
            [](const MemoryConfig& memory_config) {
                return memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED;
            },
            "Whether tensor data is interleaved across multiple DRAM channels")
        .def_readonly("buffer_type", &MemoryConfig::buffer_type, "Buffer type to store tensor data. Can be DRAM or L1")
        .def_readonly("memory_layout", &MemoryConfig::memory_layout, "Memory layout of tensor data.")
        .def_readwrite("shard_spec", &MemoryConfig::shard_spec, "Memory layout of tensor data.")
        .def(py::self == py::self)
        .def(py::self != py::self);

    m_tensor.def(
        "dump_memory_config",
        py::overload_cast<const std::string&, const MemoryConfig&>(&dump_memory_config),
        R"doc(
            Dump memory config to file
        )doc");

    m_tensor.def(
        "load_memory_config",
        py::overload_cast<const std::string&>(&load_memory_config),
        R"doc(
            Load memory config to file
        )doc");

    auto py_owned_buffer_for_uint8_t =
        py::class_<owned_buffer::Buffer<uint8_t>>(m_tensor, "owned_buffer_for_uint8_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<uint8_t>, uint8_t>(py_owned_buffer_for_uint8_t);

    auto py_owned_buffer_for_uint16_t =
        py::class_<owned_buffer::Buffer<uint16_t>>(m_tensor, "owned_buffer_for_uint16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<uint16_t>, uint16_t>(py_owned_buffer_for_uint16_t);

    auto pyCoreRange = tt_serializable_class<CoreRange>(m_tensor, "CoreRange", R"doc(
        Class defining a range of cores)doc");
    pyCoreRange.def(py::init<>([](const CoreCoord& start, const CoreCoord& end) { return CoreRange{start, end}; }))
        .def_readonly("start", &CoreRange::start_coord)
        .def_readonly("end", &CoreRange::end_coord)
        .def("grid_size", &CoreRange::grid_size);

    auto pyCoreRangeSet = tt_serializable_class<CoreRangeSet>(m_tensor, "CoreRangeSet", R"doc(
        Class defining a set of CoreRanges required for sharding)doc");
    pyCoreRangeSet.def(py::init<>([](const std::set<CoreRange>& core_ranges) { return CoreRangeSet(core_ranges); }))
        .def(
            "bounding_box",
            &CoreRangeSet::bounding_box,
            "Returns a CoreRange i.e. bounding box covering all the core ranges in the CoreRangeSet")
        .def("num_cores", &CoreRangeSet::num_cores, "Returns total number of cores in the CoreRangeSet");

    m_tensor.def(
        "num_cores_to_core_range_set",
        &num_cores_to_core_range_set,
        py::arg().noconvert(),
        py::arg().noconvert(),
        py::arg("row_wise").noconvert() = false,
        R"doc(
            Returns a CoreRangeSet from number of cores
        )doc");

    auto pyShardSpec = tt_serializable_class<ShardSpec>(m_tensor, "ShardSpec", R"doc(
        Class defining the specs required for sharding.
    )doc");

    pyShardSpec
        .def(py::init<>([](const CoreRangeSet& core_sets,
                           const std::array<uint32_t, 2>& shard_shape,
                           const ShardOrientation& shard_orientation,
                           const bool& halo) { return ShardSpec(core_sets, shard_shape, shard_orientation, halo); }))
        .def_readwrite("shape", &ShardSpec::shape, "Shape of shard.")
        .def_readwrite("grid", &ShardSpec::grid, "Grid to layout shards.")
        .def_readwrite("orientation", &ShardSpec::orientation, "Orientation of cores to read shards")
        .def("num_cores", &ShardSpec::num_cores, "Number of cores")
        .def(py::self == py::self)
        .def(py::self != py::self)
    ;

    auto py_owned_buffer_for_int32_t =
        py::class_<owned_buffer::Buffer<int32_t>>(m_tensor, "owned_buffer_for_int32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<int32_t>, int32_t>(py_owned_buffer_for_int32_t);

    auto py_owned_buffer_for_uint32_t =
        py::class_<owned_buffer::Buffer<uint32_t>>(m_tensor, "owned_buffer_for_uint32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<uint32_t>, uint32_t>(py_owned_buffer_for_uint32_t);

    auto py_owned_buffer_for_float32_t =
        py::class_<owned_buffer::Buffer<float>>(m_tensor, "owned_buffer_for_float32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<float>, float>(py_owned_buffer_for_float32_t);

    auto py_owned_buffer_for_bfloat16_t =
        py::class_<owned_buffer::Buffer<bfloat16>>(m_tensor, "owned_buffer_for_bfloat16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<bfloat16>, bfloat16>(py_owned_buffer_for_bfloat16_t);

    auto py_borrowed_buffer_for_uint8_t = py::class_<borrowed_buffer::Buffer<std::uint8_t>>(
        m_tensor, "borrowed_buffer_for_uint8_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::uint8_t>, std::uint8_t>(
        py_borrowed_buffer_for_uint8_t);

    auto py_borrowed_buffer_for_uint16_t = py::class_<borrowed_buffer::Buffer<std::uint16_t>>(
        m_tensor, "borrowed_buffer_for_uint16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::uint16_t>, std::uint16_t>(
        py_borrowed_buffer_for_uint16_t);

    auto py_borrowed_buffer_for_int32_t = py::class_<borrowed_buffer::Buffer<std::int32_t>>(
        m_tensor, "borrowed_buffer_for_int32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::int32_t>, std::int32_t>(
        py_borrowed_buffer_for_int32_t);

    auto py_borrowed_buffer_for_uint32_t = py::class_<borrowed_buffer::Buffer<std::uint32_t>>(
        m_tensor, "borrowed_buffer_for_uint32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::uint32_t>, std::uint32_t>(
        py_borrowed_buffer_for_uint32_t);

    auto py_borrowed_buffer_for_float32_t =
        py::class_<borrowed_buffer::Buffer<float>>(m_tensor, "borrowed_buffer_for_float32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<float>, float>(py_borrowed_buffer_for_float32_t);

    auto py_borrowed_buffer_for_bfloat16_t = py::class_<borrowed_buffer::Buffer<bfloat16>>(
        m_tensor, "borrowed_buffer_for_bfloat16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<bfloat16>, bfloat16>(py_borrowed_buffer_for_bfloat16_t);

    py::class_<DeviceComputeKernelConfig>(m_tensor, "DeviceComputeKernelConfig");

    py::class_<GrayskullComputeKernelConfig>(m_tensor, "GrayskullComputeKernelConfig")
        .def(
            py::init<MathFidelity, bool>(),
            py::kw_only(),
            py::arg("math_fidelity") = MathFidelity::Invalid,
            py::arg("math_approx_mode") = true)
        .def_readwrite("math_fidelity", &GrayskullComputeKernelConfig::math_fidelity)
        .def_readwrite("math_approx_mode", &GrayskullComputeKernelConfig::math_approx_mode);

    py::class_<WormholeComputeKernelConfig>(m_tensor, "WormholeComputeKernelConfig")
        .def(
            py::init<MathFidelity, bool, bool, bool>(),
            py::kw_only(),
            py::arg("math_fidelity") = MathFidelity::Invalid,
            py::arg("math_approx_mode") = true,
            py::arg("fp32_dest_acc_en") = false,
            py::arg("packer_l1_acc") = false)
        .def_readwrite("math_fidelity", &WormholeComputeKernelConfig::math_fidelity)
        .def_readwrite("math_approx_mode", &WormholeComputeKernelConfig::math_approx_mode)
        .def_readwrite("fp32_dest_acc_en", &WormholeComputeKernelConfig::fp32_dest_acc_en)
        .def_readwrite("packer_l1_acc", &WormholeComputeKernelConfig::packer_l1_acc);

    m_tensor.def(
        "layernorm_pre_allgather",
        tt::operations::primary::layernorm_pre_allgather,
        py::arg("input").noconvert(),
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        py::arg("output_dtype").noconvert() = DataType::BFLOAT16,
        R"doc(
            Performs the first part of a distributed layernorm operation collecting local statistics E(x) and E(xˆ2).
        )doc");

    m_tensor.def(
        "rmsnorm_pre_allgather",
        tt::operations::primary::rmsnorm_pre_allgather,
        py::arg("input").noconvert(),
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        py::arg("output_dtype").noconvert() = DataType::BFLOAT16,
        R"doc(
            Performs the first part of a distributed rms norm operation collecting local statistics E(x) and E(xˆ2).
        )doc");

    m_tensor.def(
        "layernorm_post_allgather",
        tt::operations::primary::layernorm_post_allgather,
        py::arg("input").noconvert(),
        py::arg("stats").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
            Performs the second part of a distributed layernorm operation normalizing the input based on the gathered statistics input.
        )doc");

    m_tensor.def(
        "rmsnorm_post_allgather",
        tt::operations::primary::rmsnorm_post_allgather,
        py::arg("input").noconvert(),
        py::arg("stats").noconvert(),
        py::arg("eps").noconvert(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
            Performs the second part of a distributed rms norm operation normalizing the input based on the gathered statistics input.
        )doc");

    m_tensor.def("fill_cache", &fill_cache,
         py::arg("cache").noconvert(), py::arg("input").noconvert(), py::arg("batch_idx"), R"doc(
        "Fills the cache tensor in place with the values from input at the specified batch_idx.
    )doc");
    m_tensor.def(
        "update_cache",
        &update_cache,
        py::arg("cache").noconvert(),
        py::arg("input").noconvert(),
        py::arg("update_idx"),
        py::arg("batch_offset") = 0,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Updates the cache tensor in place with the values from input at the specified update_idx. When cache has batch less than 32, input is assumed to have batch padded to 32 and [batch_offset:batch_offset+batch] from dim[-2] of input is used to update the cache.
    )doc");

    // TMs
    m_tensor.def(
        "convert_conv_weight_tensor_to_tiled_layout",
        &convert_conv_weight_tensor_to_tiled_layout,
        py::arg("conv_weight_tensor").noconvert(),
        py::arg("in1_block_h"),
        py::arg("in1_block_w"),
        py::arg("output_dtype").noconvert() = std::nullopt,
        R"doc(
        Converts convolution weights to 2d matrix tiled layout on host
        Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def(
        "convert_conv_weight_tensor_to_special_padding_tiled_layout",
        &convert_conv_weight_tensor_to_special_padding_tiled_layout,
        py::arg("conv_weight_tensor").noconvert(),
        py::arg("in1_block_h"),
        py::arg("in1_block_w"),
        py::arg("output_dtype").noconvert() = std::nullopt,
        R"doc(
       Converts convolution weights to 2d matrix tiled layout on host with special block height padding
       Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def(
        "convert_conv_weight_tensor_to_grouped_layout",
        &convert_conv_weight_tensor_to_grouped_layout,
        py::arg("conv_weight_tensor").noconvert(),
        py::arg("num_groups"),
        py::arg("output_dtype").noconvert() = std::nullopt,
        R"doc(
        Converts convolution weights to grouped layout with padded zeros
        Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def(
        "format_input_tensor",
        &AutoFormat::format_input_tensor,
        py::arg("input").noconvert(),
        py::arg("device").noconvert(),
        py::arg("padded_shape"),
        py::arg("pad_value"),
        py::arg("target_layout").noconvert(),
        py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
            Formats tensor to target layout and pads to padded shape
        )doc");
    m_tensor.def(
        "format_output_tensor",
        &AutoFormat::format_output_tensor,
        py::arg("output").noconvert(),
        py::arg("shape"),
        py::arg("device").noconvert(),
        py::arg("target_layout").noconvert(),
        py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
            Formats tensor to target layout and unpads to shape
        )doc");
    m_tensor.def(
        "pad_to_tile_shape",
        [](const std::array<uint32_t, 4>& unpadded_shape,
           bool pad_c = false,
           bool pad_n = false,
           bool pad_h = true,
           bool pad_w = true) -> Shape {
            return AutoFormat::pad_to_tile_shape(unpadded_shape, pad_c, pad_n, pad_h, pad_w);
        },
        R"doc(
            Returns shape padded to tile shape
        )doc");
    m_tensor.def(
        "dump_tensor",
        py::overload_cast<const std::string&, const Tensor&, const std::unordered_map<std::string, std::string>&>(&tt::tt_metal::dump_tensor),
        py::arg("filename"),
        py::arg("tensor"),
        py::arg("strategy") = std::unordered_map<std::string, std::string>{},
        R"doc(
            Dump tensor to file
        )doc");

    m_tensor.def(
        "dump_tensor",
        py::overload_cast<std::ostream&, const Tensor&, const std::unordered_map<std::string, std::string>&>(&tt::tt_metal::dump_tensor),
        py::arg("output_stream"),
        py::arg("tensor"),
        py::arg("strategy") = std::unordered_map<std::string, std::string>{},
        R"doc(
            Dump tensor to output stream
        )doc");

    m_tensor.def(
        "load_tensor",
        py::overload_cast<const std::string&, Device*>(&tt::tt_metal::load_tensor<Device*>),
        py::arg("file_name"),
        py::arg("device") = nullptr,
        R"doc(Load tensor from file to Device)doc");

    m_tensor.def(
        "load_tensor",
        py::overload_cast<const std::string&, DeviceMesh*>(&tt::tt_metal::load_tensor<DeviceMesh*>),
        py::arg("file_name"),
        py::arg("device") = nullptr,
        R"doc(Load tensor from file to DeviceMesh)doc");

    m_tensor.def(
        "load_tensor",
        py::overload_cast<std::istream&, Device*>(&tt::tt_metal::load_tensor<Device*>),
        py::arg("input_stream"),
        py::arg("device") = nullptr,
        R"doc(Load tensor from input stream to Device)doc");

    m_tensor.def(
        "load_tensor",
        py::overload_cast<std::istream&, DeviceMesh*>(&tt::tt_metal::load_tensor<DeviceMesh*>),
        py::arg("input_stream"),
        py::arg("device") = nullptr,
        R"doc(Load tensor from input stream to DeviceMesh)doc");

    m_tensor.def(
        "num_cores_to_corerange_set",
        py::overload_cast<const uint32_t, const CoreCoord, const bool>(&num_cores_to_corerange_set),
        R"doc(
            Create a CoreRangeSet containing the specified number of cores
        )doc");

    m_tensor.def(
        "allocate_tensor_on_device",
        py::overload_cast<const ttnn::Shape&, DataType, Layout, Device*, const MemoryConfig&>(
            &allocate_tensor_on_device),
        py::arg("shape"),
        py::arg("dtype"),
        py::arg("layout"),
        py::arg("device"),
        py::arg("memory_config") = MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED},
        R"doc(
            Allocate a tensor with specified attributes on a device.
        )doc");

    m_tensor.def(
        "allocate_tensor_on_device",
        py::overload_cast<const ttnn::Shape&, DataType, Layout, DeviceMesh*, const MemoryConfig&>(
            &allocate_tensor_on_device),
        py::arg("shape"),
        py::arg("dtype"),
        py::arg("layout"),
        py::arg("device"),
        py::arg("memory_config") = MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED},
        R"doc(
            Allocate a tensor with specified attributes on a device.
        )doc");

    m_tensor.def(
        "write_tensor",
        py::overload_cast<Tensor, Tensor, uint8_t>(&write_tensor),
        py::arg("host_tensor"),
        py::arg("device_tensor"),
        py::arg("cq_id") = 0,
        R"doc(
            Copy a host tensor to its equivalent tensor on a device.
        )doc");

    detail::TensorModulePyTensor(m_tensor);
    detail::TensorModuleDMOPs(m_tensor);
}

}  // namespace tt::tt_metal
