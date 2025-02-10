// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/distributed_pybind.hpp"
#include <pybind11/pytypes.h>
#include <utility>

#include "tt-metalium/assert.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/command_queue.hpp>
#include "pybind11/stl.h"

using namespace tt::tt_metal;

namespace ttnn::distributed {

namespace py = pybind11;

py::object get_torch_type(DataType& dtype, const py::object& torch) {
    if (dtype == DataType::UINT8) {
        return torch.attr("uint8");
    } else if (dtype == DataType::UINT16) {
        return torch.attr("int16");
    } else if (dtype == DataType::INT32) {
        return torch.attr("int32");
    } else if (dtype == DataType::UINT32) {
        return torch.attr("int32");
    } else if (dtype == DataType::FLOAT32) {
        return torch.attr("float32");
    } else if (dtype == DataType::BFLOAT16) {
        return torch.attr("bfloat16");
    }
    TT_THROW("Unsupported DataType: {}", dtype);
}

// duplicated from pytensor.cpp
template <typename T>
owned_buffer::Buffer<T> create_row_major_owned_buffer(
    owned_buffer::Buffer<T>&& owned_buffer, const ttnn::TensorSpec& tensor_spec, const bool legacy_output) {
    TT_FATAL(
        !tensor_spec.memory_config().is_sharded() or tensor_spec.memory_config().shard_spec.has_value(),
        "Sharded tensors must have a shard spec when converting to tt tensors!");

    if (legacy_output) {
        if (tensor_spec.layout() == Layout::TILE) {
            auto data = tensor_impl::convert_layout_tile_to_row_major(
                tensor_spec.physical_shape(), tensor_spec.tile(), owned_buffer);
            return owned_buffer::create(std::move(data));
        }
        return owned_buffer;
    }

    auto physical_data = owned_buffer.get();

    // See implementation for documentation
    auto logical_data = tensor_impl::decode_tensor_data(std::move(physical_data), tensor_spec);

    return owned_buffer::create(std::move(logical_data));
}

// duplicated from pytensor.cpp
OwnedBuffer get_host_buffer_from_tensor(const Tensor& tt_tensor, const bool legacy_output) {
    TT_ASSERT(tt_tensor.storage_type() == StorageType::OWNED);

    OwnedStorage storage = std::visit(
        [&tt_tensor](auto&&) -> OwnedBuffer {
            TT_THROW(
                "Tensor with {} cannot be converted to torch",
                tt::stl::get_active_type_name_in_variant(tt_tensor.get_storage()));
        },
        tt_tensor.get_storage());

    const auto& tensor_spec = tt_tensor.get_tensor_spec();
    const auto tt_dtype = tensor_spec.data_type();
    switch (tt_dtype) {
        case DataType::UINT8: {
            return create_row_major_owned_buffer(
                std::move(owned_buffer::get_as<uint8_t>(storage.buffer)), tensor_spec, legacy_output);
        }
        case DataType::UINT16: {
            return create_row_major_owned_buffer(
                std::move(owned_buffer::get_as<uint16_t>(storage.buffer)), tensor_spec, legacy_output);
        }
        case DataType::INT32: {
            return create_row_major_owned_buffer(
                std::move(owned_buffer::get_as<int32_t>(storage.buffer)), tensor_spec, legacy_output);
        }
        case DataType::UINT32: {
            return create_row_major_owned_buffer(
                std::move(owned_buffer::get_as<uint32_t>(storage.buffer)), tensor_spec, legacy_output);
        }
        case DataType::FLOAT32: {
            return create_row_major_owned_buffer(
                std::move(owned_buffer::get_as<float>(storage.buffer)), tensor_spec, legacy_output);
        }
        case DataType::BFLOAT16: {
            return create_row_major_owned_buffer(
                std::move(owned_buffer::get_as<::bfloat16>(storage.buffer)), tensor_spec, legacy_output);
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            const auto& tile = tensor_spec.tile();
            auto uint32_data = owned_buffer::get_as<std::uint32_t>(storage.buffer).get();
            auto float_unpacked_data = tt_dtype == DataType::BFLOAT8_B
                                           ? unpack_bfp8_tiles_into_float_vec(
                                                 uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                                           : unpack_bfp4_tiles_into_float_vec(
                                                 uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
            auto input_float_buffer = owned_buffer::create<float>(std::move(float_unpacked_data));
            return create_row_major_owned_buffer(std::move(input_float_buffer), tensor_spec, legacy_output);
        }
        default: {
            TT_THROW("Unsupported DataType: {}", tt_dtype);
            break;
        }
    }
}

// duplicated from pytensor.cpp
py::object convert_tt_tensor_to_torch_tensor(const Tensor& tt_tensor, const bool legacy_output = false) {
    // TODO: Remove legacy_output flag which supports old behaviour of returning tensors with padded shape.
    // These cases need to be fixed:
    //     ROW_MAJOR tensors with padding (since ROW_MAJOR has no alignment, cannot automatically strip data unless
    //     padded shape is queried) Physical sharding on padded shape (unlike interleaved tensors, cannot derive an
    //     equivalent logical shard spec to strip out data)
    // One way to clean this up is:
    //     1. Update tests to use ttnn.from_torch and ttnn.to_torch
    //     2. Fix usage of tensor.to_torch inside ttnn functional APIs
    //     3. Deprecate old tensor.to_torch and rename tensor.to_torch_with_logical_shape back to tensor.to_torch
    auto buffer = get_host_buffer_from_tensor(tt_tensor, legacy_output);

    py::object torch = py::module_::import("torch");
    auto frombuffer = torch.attr("frombuffer");

    DataType dtype = tt_tensor.get_tensor_spec().data_type();

    py::object torch_dtype = get_torch_type(dtype, torch);

    auto logical_shape = tt_tensor.get_logical_shape();
    auto view = logical_shape.view();
    std::vector<uint32_t> torch_shape(view.begin(), view.end());
    auto tensor = [&]() {
        if (tt_tensor.volume() == 0) {
            auto pytorch_empty = torch.attr("empty");
            return pytorch_empty(torch_shape, py::arg("dtype") = torch_dtype);
        }
        return frombuffer(buffer, py::arg("dtype") = torch_dtype);
    }();

    if (legacy_output) {
        auto shape = tt_tensor.get_padded_shape();
        torch_shape = std::vector<std::uint32_t>(shape.cbegin(), shape.cend());
    }
    tensor = tensor.attr("reshape")(torch_shape);
    tensor = tensor.attr("contiguous")();
    return tensor;
}

void py_module_types(py::module& module) {
    py::class_<MeshDevice, std::shared_ptr<MeshDevice>>(module, "MeshDevice");
    py::class_<MeshSubDeviceManagerId>(module, "MeshSubDeviceManagerId");
    py::class_<MeshShape>(module, "MeshShape", "Struct representing the shape of a mesh device.");
    py::class_<MeshOffset>(module, "MeshOffset", "Struct representing the offset of a mesh device.");
}

void py_module(py::module& module) {
    static_cast<py::class_<MeshShape>>(module.attr("MeshShape"))
        .def(
            py::init([](size_t num_rows, size_t num_cols) { return MeshShape(num_rows, num_cols); }),
            "Constructor with specified number of rows and columns.",
            py::arg("num_rows"),
            py::arg("num_cols"))
        .def_readwrite("num_rows", &MeshShape::num_rows, "Number of rows in the mesh.")
        .def_readwrite("num_cols", &MeshShape::num_cols, "Number of columns in the mesh.")
        .def(
            "__repr__",
            [](const MeshShape& ms) {
                return "<MeshShape num_rows=" + std::to_string(ms.num_rows) +
                       " num_cols=" + std::to_string(ms.num_cols) + ">";
            })
        .def("__iter__", [](const MeshShape& ms) { return py::iter(py::make_tuple(ms.num_rows, ms.num_cols)); });
    static_cast<py::class_<MeshOffset>>(module.attr("MeshOffset"))
        .def(
            py::init([](size_t row, size_t col) { return MeshOffset(row, col); }),
            "Constructor with specified row and column offsets.",
            py::arg("row"),
            py::arg("col"))
        .def_readwrite("row", &MeshOffset::row, "Row offset in the mesh.")
        .def_readwrite("col", &MeshOffset::col, "Column offset in the mesh.")
        .def(
            "__repr__",
            [](const MeshOffset& mo) {
                return "<MeshOffset row=" + std::to_string(mo.row) + " col=" + std::to_string(mo.col) + ">";
            })
        .def("__iter__", [](const MeshOffset& mo) { return py::iter(py::make_tuple(mo.row, mo.col)); });

    auto py_mesh_device = static_cast<py::class_<MeshDevice, std::shared_ptr<MeshDevice>>>(module.attr("MeshDevice"));
    py_mesh_device
        .def(
            py::init([](const MeshShape& mesh_device_shape,
                        size_t l1_small_size,
                        size_t trace_region_size,
                        size_t num_command_queues,
                        const DispatchCoreConfig& dispatch_core_config,
                        const MeshOffset& offset,
                        const std::vector<chip_id_t>& physical_device_ids) {
                return MeshDevice::create(
                    MeshDeviceConfig{
                        .mesh_shape = mesh_device_shape,
                        .offset = offset,
                        .physical_device_ids = physical_device_ids,
                    },
                    l1_small_size,
                    trace_region_size,
                    num_command_queues,
                    dispatch_core_config);
            }),
            py::kw_only(),
            py::arg("mesh_shape"),
            py::arg("l1_small_size"),
            py::arg("trace_region_size"),
            py::arg("num_command_queues"),
            py::arg("dispatch_core_config"),
            py::arg("offset"),
            py::arg("physical_device_ids"))
        .def("get_num_devices", &MeshDevice::num_devices)
        .def("id", &MeshDevice::id)
        .def("get_device_ids", &MeshDevice::get_device_ids)
        .def(
            "get_device",
            py::overload_cast<chip_id_t>(&MeshDevice::get_device, py::const_),
            py::return_value_policy::reference)
        .def(
            "get_device",
            py::overload_cast<size_t, size_t>(&MeshDevice::get_device, py::const_),
            py::return_value_policy::reference)
        .def(
            "get_devices",
            &MeshDevice::get_devices,
            py::return_value_policy::reference,
            R"doc(
            Get the devices in the device mesh.

            Returns:
                List[Device]: The devices in the device mesh.
        )doc")
        .def(
            "create_submesh",
            &MeshDevice::create_submesh,
            py::arg("submesh_shape"),
            py::arg("offset"),
            py::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevice is alive
        .def(
            "create_submeshes",
            &MeshDevice::create_submeshes,
            py::arg("submesh_shape"),
            py::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevices are alive
        .def(
            "compute_with_storage_grid_size",
            &MeshDevice::compute_with_storage_grid_size,
            R"doc(
            Get the compute grid size (x, y) of the first device in the device mesh denoting region that can be targeted by ops.

            Returns:
                CoreCoord: The compute grid size of the first device in the device mesh.
        )doc")
        .def(
            "dram_grid_size",
            &MeshDevice::dram_grid_size,
            R"doc(
            Get the dram grid size (x, y) of the first device in the device mesh.

            Returns:
                CoreCoord: The dram grid size of the first device in the device mesh.
        )doc")
        .def(
            "arch",
            &MeshDevice::arch,
            R"doc(
            Get the arch of the first device in the device mesh.

            Returns:
                Arch: The arch of the first device in the device mesh.
        )doc")
        .def(
            "enable_async",
            &MeshDevice::enable_async,
            py::arg("enable"),
            R"doc(
                Enable or disable async mode across all devices in the mesh.

                Args:
                    enable (bool): True to enable async mode, False to disable it.
            )doc")
        .def(
            "enable_program_cache",
            &MeshDevice::enable_program_cache,
            R"doc(
                Enable program cache across all devices in the mesh.
            )doc")
        .def(
            "disable_and_clear_program_cache",
            &MeshDevice::disable_and_clear_program_cache,
            R"doc(
                Disable program cache across all devices in the mesh.
            )doc")
        .def_property_readonly(
            "shape",
            &MeshDevice::shape,
            R"doc(
            Get the shape of the device mesh.

            Returns:
                Tuple[int, int]: The shape of the device mesh as (num_rows, num_cols).
        )doc")
        .def(
            "reshape",
            &MeshDevice::reshape,
            py::arg("new_shape"),
            R"doc(
                Reshapes the logical mesh and re-maps the physical devices to the new logical coordinates.

                Reshaping Rules:
                1. The old_shape volume must equal the new_shape volume (i.e. number of devices must remain constant)
                2. Line-to-Line Reshaping (when either dimension is 1):
                   - Always possible between 1xN and Nx1 shapes (e.g.: 1x8 <-> 8x1)
                3. Grid-to-Grid Reshaping:
                   - Only possible if the devices can form a connected physical mesh in the new shape
                   - Must maintain physical connectivity between adjacent devices
                4. Line-to-Grid Reshaping:
                   - Only possible if the physical devices can form a connected physical mesh in the new shape
                   - Example: 1x8 -> 2x4 is possible only if physical mesh permits a 2x4 configuration

                Args:
                    new_shape (MeshShape): The new shape of the mesh.

                Raises:
                    RuntimeError: If the reshaping constraints are not met:
                    1. The old_shape volume must equal the new_shape volume (i.e. number of devices must remain constant)
                    2. For Grid-to-Grid or Line-to-Grid reshaping: physical connectivity must be possible with current devices
            )doc")
        .def("__repr__", &MeshDevice::to_string)
        .def(
            "create_sub_device_manager",
            [](MeshDevice& self, const std::vector<SubDevice>& sub_devices, DeviceAddr local_l1_size) {
                return self.mesh_create_sub_device_manager(sub_devices, local_l1_size);
            },
            py::arg("sub_devices"),
            py::arg("local_l1_size"),
            R"doc(
                Creates a sub-device manager for the given mesh device.

                Args:
                    sub_devices (List[ttnn.SubDevice]): The sub-devices to include in the sub-device manager.
                    This configuration will be used for each device in the MeshDevice.
                    local_l1_size (int): The size of the local allocators of each sub-device. The global allocator will be shrunk by this amount.

                Returns:
                    MeshSubDeviceManagerId: The ID of the created sub-device manager.
            )doc")
        .def(
            "create_sub_device_manager_with_fabric",
            [](MeshDevice& self, const std::vector<SubDevice>& sub_devices, DeviceAddr local_l1_size) {
                return self.mesh_create_sub_device_manager_with_fabric(sub_devices, local_l1_size);
            },
            py::arg("sub_devices"),
            py::arg("local_l1_size"),
            R"doc(
                Creates a sub-device manager for the given mesh device. This will automatically create a sub-device of ethernet cores for use with fabric.
                Note that this is a temporary API until migration to actual fabric is complete.

                Args:
                    sub_devices (List[ttnn.SubDevice]): The sub-devices to include in the sub-device manager. No ethernet cores should be included in this list.
                    This configuration will be used for each device in the MeshDevice.
                    local_l1_size (int): The size of the local allocators of each sub-device. The global allocator will be shrunk by this amount.

                Returns:
                    MeshSubDeviceManagerId: The ID of the created sub-device manager.
                    SubDeviceId: The ID of the sub-device that will be used for fabric.
            )doc")
        .def(
            "load_sub_device_manager",
            &MeshDevice::mesh_load_sub_device_manager,
            py::arg("mesh_sub_device_manager_id"),
            R"doc(
                Loads the sub-device manager with the given ID.

                Args:
                    mesh_sub_device_manager_id (MeshSubDeviceManagerId): The ID of the sub-device manager to load.
            )doc")
        .def(
            "clear_loaded_sub_device_manager",
            &MeshDevice::mesh_clear_loaded_sub_device_manager,
            R"doc(
                Clears the loaded sub-device manager for the given mesh device.
            )doc")
        .def(
            "remove_sub_device_manager",
            &MeshDevice::mesh_remove_sub_device_manager,
            py::arg("mesh_sub_device_manager_id"),
            R"doc(
                Removes the sub-device manager with the given ID.

                Args:
                    mesh_sub_device_manager_id (MeshSubDeviceManagerId): The ID of the sub-device manager to remove.
            )doc")
        .def(
            "set_sub_device_stall_group",
            [](MeshDevice& self, const std::vector<SubDeviceId>& sub_device_ids) {
                self.mesh_set_sub_device_stall_group(sub_device_ids);
            },
            py::arg("sub_device_ids"),
            R"doc(
                Set the SubDevice IDs that will be stalled on by default for Fast Dispatch commands such as reading, writing, synchronizing.
                Stalling here refers to the Fast Dispatch cores waiting for programs to complete execution on the specified SubDevices before proceeding with the specified instruction.
                The default SubDevice IDs to stall on are set to all SubDevice IDs, and whenever a new SubDevice Manager is loaded.

                Args:
                    sub_device_ids (List[SubDeviceId]): The IDs of the SubDevices to stall on.
            )doc")
        .def(
            "reset_sub_device_stall_group",
            &MeshDevice::mesh_reset_sub_device_stall_group,
            R"doc(
                Resets the sub_device_ids that will be stalled on by default for Fast Dispatch commands such as reading, writing, synchronizing
                back to all SubDevice IDs.
            )doc");

    module.def(
        "open_mesh_device",
        &open_mesh_device,
        py::kw_only(),
        py::arg("mesh_shape"),
        py::arg("l1_small_size"),
        py::arg("trace_region_size"),
        py::arg("num_command_queues"),
        py::arg("offset"),
        py::arg("physical_device_ids"),
        py::arg("dispatch_core_config"));

    module.def("close_mesh_device", &close_mesh_device, py::arg("mesh_device"), py::kw_only());
    module.def(
        "get_device_tensor",
        py::overload_cast<const Tensor&, int>(&ttnn::distributed::get_device_tensor),
        py::arg("tensor"),
        py::arg("device_id"),
        py::kw_only(),
        R"doc(
        Get the tensor shard corresponding to the device_id.

        Args:
            tensor (Tensor): The tensor to get the shard from.
            device_id (int): The device id to get the shard for.

        Returns:
            Tensor: The shard of the tensor corresponding to the device_id.
    )doc");
    module.def(
        "get_device_tensor",
        py::overload_cast<const Tensor&, const IDevice*>(&ttnn::distributed::get_device_tensor),
        py::arg("tensor"),
        py::arg("device"),
        py::kw_only(),
        R"doc(
        Get the tensor shard corresponding to the device.

        Args:
            tensor (Tensor): The tensor to get the shard from.
            device (Device): The device to get the shard for.

        Returns:
            Tensor: The shard of the tensor corresponding to the device.
    )doc");
    module.def("get_device_tensors", &get_device_tensors, py::arg("tensor"), py::kw_only());
    module.def(
        "aggregate_as_tensor",
        [](const std::vector<Tensor>& tensors) -> Tensor { return aggregate_as_tensor(tensors, AllGatherTensor{}); },
        py::arg("tensors"),
        py::kw_only());
    module.def(
        "shardedtensor_to_tensorlist",
        [](const Tensor& tensor) -> std::vector<py::object> {
            std::vector<py::object> tensorlist_local;

            std::vector<ttnn::Tensor> tensors = get_device_tensors(tensor);

            tensorlist_local.reserve(tensors.size());

            for (const Tensor& shard : tensors) {
                tensorlist_local.push_back(convert_tt_tensor_to_torch_tensor(shard));
            }

            return tensorlist_local;
        },
        py::arg("tensor"));
    module.def("get_t3k_physical_device_ids_ring", &get_t3k_physical_device_ids_ring);
}

}  // namespace ttnn::distributed
