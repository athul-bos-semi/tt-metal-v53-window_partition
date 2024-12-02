// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"

TEST(TensorUtilsTest, TestFloatToFromTensorEven) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> test_data = {1.F, 5.F, 10.F, 15.F};

    auto shape = ttml::core::create_shape({1, 1, 1, 4});
    auto tensor = ttml::core::from_vector(test_data, shape, device);

    auto vec_back = ttml::core::to_vector(tensor);

    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_EQ(vec_back[i], test_data[i]);
    }
}

TEST(TensorUtilsTest, TestFloatToFromTensorOdd) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> test_data = {30.F, 20.F, 2.F};

    auto shape = ttml::core::create_shape({1, 1, 1, 3});
    auto tensor = ttml::core::from_vector(test_data, shape, device);

    auto vec_back = ttml::core::to_vector(tensor);

    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_EQ(vec_back[i], test_data[i]);
    }
}

TEST(TensorUtilsTest, TestUint32ToFromTensorEven) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<uint32_t> test_data = {1, 5, 10, 15};

    auto shape = ttml::core::create_shape({1, 1, 1, 4});
    auto tensor = ttml::core::from_vector<uint32_t, DataType::UINT32>(test_data, shape, device);

    auto vec_back = ttml::core::to_vector<uint32_t>(tensor);

    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_EQ(vec_back[i], test_data[i]);
    }
}

TEST(TensorUtilsTest, TestUint32ToFromTensorOdd) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<uint32_t> test_data = {30, 20, 2};

    auto shape = ttml::core::create_shape({1, 1, 1, 3});
    auto tensor = ttml::core::from_vector<uint32_t, DataType::UINT32>(test_data, shape, device);

    auto vec_back = ttml::core::to_vector<uint32_t>(tensor);

    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_EQ(vec_back[i], test_data[i]);
    }
}

TEST(TensorUtilsTest, TestUint32ToFromTensorLargeWithBatch) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<uint32_t> test_data;
    uint32_t batch_size = 16;
    uint32_t vec_size = 256 * batch_size;
    for (size_t i = 0; i < vec_size; i++) {
        test_data.push_back(i);
    }

    auto shape = ttml::core::create_shape({batch_size, 1, 1, vec_size / batch_size});
    auto tensor = ttml::core::from_vector<uint32_t, DataType::UINT32>(test_data, shape, device);
    auto vec_back = ttml::core::to_vector<uint32_t>(tensor);
    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_EQ(vec_back[i], test_data[i]);
    }
}

TEST(TensorUtilsTest, TestFloatToFromTensorLargeWithBatch) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> test_data;
    uint32_t batch_size = 16;
    uint32_t vec_size = 256 * batch_size;
    for (size_t i = 0; i < vec_size; i++) {
        test_data.push_back((float)i / 100.0F);
    }

    auto shape = ttml::core::create_shape({batch_size, 1, 1, vec_size / batch_size});
    auto tensor = ttml::core::from_vector(test_data, shape, device);
    auto vec_back = ttml::core::to_vector(tensor);
    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_NEAR(vec_back[i], test_data[i], 0.5F);
    }
}

TEST(TensorUtilsTest, TestToFromTensorLarge) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> test_data;
    uint32_t vec_size = 1337;
    for (size_t i = 0; i < vec_size; i++) {
        test_data.push_back((float)i / 100.0F);
    }

    auto shape = ttml::core::create_shape({1, 1, 1, vec_size});
    auto tensor = ttml::core::from_vector(test_data, shape, device);
    auto vec_back = ttml::core::to_vector(tensor);
    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_NEAR(vec_back[i], test_data[i], 0.1F);
    }
}

TEST(TensorUtilsTest, TestToFromTensorBatch) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> test_data = {1.F, 5.F, 10.F, 15.F};

    auto shape = ttml::core::create_shape({2, 1, 1, 2});
    auto tensor = ttml::core::from_vector(test_data, shape, device);

    auto vec_back = ttml::core::to_vector(tensor);

    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_EQ(vec_back[i], test_data[i]);
    }
}

TEST(TensorUtilsTest, TestOnes_0) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttml::core::create_shape({1, 2, 3, 4});
    auto tensor = ttml::core::ones(shape, device);
    auto tensor_vec = ttml::core::to_vector(tensor);
    for (auto& val : tensor_vec) {
        EXPECT_EQ(val, 1.F);
    }

    auto tensor1 = ttml::core::ones(shape, device);
    auto tensor_vec1 = ttml::core::to_vector(tensor1);
    for (auto& val : tensor_vec1) {
        EXPECT_EQ(val, 1.F);
    }
}

TEST(TensorUtilsTest, TestOnes_1) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttml::core::create_shape({1, 2, 3, 4});
    auto tensor_zeros = ttml::core::zeros(shape, device);
    auto tensor_ones = ttml::core::ones(tensor_zeros.get_shape(), device);
    auto tensor_vec = ttml::core::to_vector(tensor_ones);
    for (auto& val : tensor_vec) {
        EXPECT_EQ(val, 1.F);
    }
}

TEST(TensorUtilsTest, TestZeros) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttml::core::create_shape({1, 2, 3, 4});
    auto tensor = ttml::core::ones(shape, device);

    auto zeros_like_tensor = ttml::core::zeros_like(tensor);
    auto zeros_like_tensor_vec = ttml::core::to_vector(zeros_like_tensor);
    for (auto& val : zeros_like_tensor_vec) {
        EXPECT_EQ(val, 0.F);
    }
}

TEST(TensorUtilsTest, TestIsInitialized) {
    auto* device = &ttml::autograd::ctx().get_device();

    tt::tt_metal::Tensor tensor;
    EXPECT_FALSE(ttml::core::is_tensor_initialized(tensor));

    auto shape = ttml::core::create_shape({1, 2, 3, 4});
    tensor = ttml::core::zeros(shape, device);
    EXPECT_TRUE(ttml::core::is_tensor_initialized(tensor));
}

TEST(TensorUtilsTest, TestOnesLike) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttml::core::create_shape({1, 2, 32, 321});
    auto tensor_zeros = ttml::core::zeros(shape, device);
    auto tensor_ones = ttml::core::ones_like(tensor_zeros);
    auto tensor_vec = ttml::core::to_vector(tensor_ones);
    for (auto& val : tensor_vec) {
        EXPECT_EQ(val, 1.F);
    }
}

TEST(TensorUtilsTest, TestZerosLike) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttml::core::create_shape({1, 2, 31, 322});
    auto tensor_ones = ttml::core::ones(shape, device);
    auto tensor_zeros = ttml::core::zeros_like(tensor_ones);
    auto tensor_vec = ttml::core::to_vector(tensor_zeros);
    for (auto& val : tensor_vec) {
        EXPECT_EQ(val, 0.F);
    }
}
