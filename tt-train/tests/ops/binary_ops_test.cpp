// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/binary_ops.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"

class BinaryOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(BinaryOpsTest, TensorAdd_Broadcasted) {
    xt::xarray<float> a = {{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}};
    xt::xarray<float> b = xt::xarray<float>::from_shape({1, 1, 1, 1});
    b(0, 0, 0, 0) = 1.F;

    auto a_tensor = ttml::autograd::create_tensor(ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device()));
    auto b_tensor = ttml::autograd::create_tensor(ttml::core::from_xtensor(b, &ttml::autograd::ctx().get_device()));

    auto result = ttml::ops::add(a_tensor, b_tensor);
    auto result_xarray = ttml::core::to_xtensor(result->get_value());

    auto expected_result = xt::xarray<float>{2.F, 3.F, 4.F, 5.F, 2.F, 3.F, 4.F, 5.F};

    EXPECT_TRUE(xt::allclose(result_xarray, expected_result));
}

TEST_F(BinaryOpsTest, TensorMul_Eltwise) {
    xt::xarray<float> a = {{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}};
    xt::xarray<float> b = {{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}};

    auto a_tensor = ttml::autograd::create_tensor(ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device()));
    auto b_tensor = ttml::autograd::create_tensor(ttml::core::from_xtensor(b, &ttml::autograd::ctx().get_device()));

    auto result = ttml::ops::mul(a_tensor, b_tensor);
    auto result_xarray = ttml::core::to_xtensor(result->get_value());

    auto expected_result = xt::xarray<float>{{1.F, 4.F, 9.F, 16.F, 1.F, 4.F, 9.F, 16.F}};

    EXPECT_TRUE(xt::allclose(result_xarray, expected_result));
}

TEST_F(BinaryOpsTest, TensorDivByFloat) {
    xt::xarray<float> a = {{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}};

    auto a_tensor = ttml::autograd::create_tensor(ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device()));
    float b = 2.F;
    auto result = ttml::ops::div(a_tensor, b);
    auto result_xarray = ttml::core::to_xtensor(result->get_value());

    auto expected_result = xt::xarray<float>{{0.5F, 1.F, 1.5F, 2.F, 0.5F, 1.F, 1.5F, 2.F}};

    EXPECT_TRUE(xt::allclose(result_xarray, expected_result));
}
