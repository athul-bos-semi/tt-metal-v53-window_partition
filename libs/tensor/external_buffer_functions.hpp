#pragma once

#include "tensor/tensor.hpp"
#include "tensor/external_buffer.hpp"

#include <vector>

namespace tt {

namespace tt_metal {

namespace external_buffer {

template<typename T>
void validate_datatype(const Tensor& tensor) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        TT_ASSERT(tensor.dtype() == DataType::UINT32);
    } else if constexpr (std::is_same_v<T, float>) {
        TT_ASSERT(tensor.dtype() == DataType::FLOAT32 or tensor.dtype() == DataType::BFLOAT8_B);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        TT_ASSERT(tensor.dtype() == DataType::BFLOAT16);
    }
}

template<typename T>
Buffer<T> get_as(ExternalBuffer& buffer) {
    return std::get<Buffer<T>>(buffer);
}

template<typename T>
const Buffer<T> get_as(const ExternalBuffer& buffer) {
    return std::get<Buffer<T>>(buffer);
}

template<typename T>
Buffer<T> get_as(Tensor& tensor) {
    validate_datatype<T>(tensor);
    return std::visit(
        [&] (auto&& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, ExternalStorage>) {
                return get_as<T>(storage.buffer);
            } else {
                TT_THROW("Must be an external storage");
            }
        },
        tensor.storage()
    );
}

template<typename T>
const Buffer<T> get_as(const Tensor& tensor) {
    validate_datatype<T>(tensor);
    return std::visit(
        [] (auto&& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, ExternalStorage>) {
                return get_as<T>(storage.buffer);
            } else {
                TT_THROW("Must be an external storage");
            }
        },
        tensor.storage()
    );
}

}  // namespace external_buffer

}  // namespace tt_metal

}  // namespace tt
