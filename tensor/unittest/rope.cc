//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/function.h>

// FIXME IFDEF??
/// Base
#include <grid/tensor/base/rope.h>
#include <grid/tensor/base/tensor.h>
#include "tensor_base.h"

/// Metal
#if 0
#include <grid/tensor/metal/rope.h>
#include <grid/tensor/metal/tensor.h>
#include "tensor_metal.h"
#endif

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ElementsAre;

template <typename T> class RopeTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(RopeTestSuite);

TYPED_TEST_P(RopeTestSuite, TensorRopeTest)
{
  typename TypeParam::Tensor tensor = grid::Tensor{ 5, 3, 1, 9, 3, 2, 56, 1, 34, 52, 65, 98, 13 };
  typename TypeParam::Tensor result = Rope(tensor, 1);
  std::cout << result << '\n';
  //EXPECT_EQ(neg, grid::Tensor{ -5 });
}

REGISTER_TYPED_TEST_SUITE_P(RopeTestSuite,
    TensorRopeTest
);

INSTANTIATE_TYPED_TEST_SUITE_P(RopeTestBase, RopeTestSuite, TensorBaseType);
// IFDEF METAL
#if 0
INSTANTIATE_TYPED_TEST_SUITE_P(MetalTestBase, RopeTestSuite, TensorMetalType);
#endif
