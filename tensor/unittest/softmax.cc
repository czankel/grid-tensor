//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/generator.h>
#include <grid/tensor/precision.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <grid/tensor/base/tensor.h>
#include <grid/tensor/base/generator.h>
#include <grid/tensor/base/softmax.h>
#include "tensor_base.h"

#ifdef BUILD_METAL
#include <grid/tensor/metal/tensor.h>
#include <grid/tensor/metal/softmax.h>
#include "tensor_metal.h"
#endif

#ifdef BUILD_CUDA
#include <grid/tensor/cuda/tensor.h>
#include <grid/tensor/cuda/softmax.h>
#include "tensor_cuda.h"
#endif


using testing::ElementsAre;


template <typename T> class SoftMaxTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(SoftMaxTestSuite);


TYPED_TEST_P(SoftMaxTestSuite, TensorSoftMaxRank1)
{
  typename TypeParam::Tensor tensor =
    grid::Tensor{ 1.618f, 2.f, 3.14f, 5.382f, -8.5f, 1.3f, -2.1f, 3.477f, 5.5f };
  grid::Tensor expected {
    0.944665670912764495e-02f, 0.138413555743161758e-01f, 0.432787127410648800e-01f,
    0.407345162648720671e-00f, 0.381141786703997885e-06f, 0.687341376275671639e-02f,
    0.229388293053500676e-03f, 0.606221835647282256e-01f, 0.458362745564445395e-00f };

  grid::Precision p(100.f);
  typename TypeParam::Tensor result = grid::SoftMax(tensor);
  EXPECT_EQ(result, expected);
}


TYPED_TEST_P(SoftMaxTestSuite, TensorSoftMaxRank1Large)
{
  grid::Precision p(100.f);
  auto random = grid::Random<grid::Tensor, float>({10000,7000})();

  typename TypeParam::Tensor tensor{random};
  typename TypeParam::Tensor result = grid::SoftMax(tensor);

  grid::Tensor expected = grid::SoftMax(random);
  EXPECT_EQ(result, expected);
}


REGISTER_TYPED_TEST_SUITE_P(SoftMaxTestSuite,
    TensorSoftMaxRank1,
    TensorSoftMaxRank1Large);

INSTANTIATE_TYPED_TEST_SUITE_P(SoftMaxTestBase, SoftMaxTestSuite, TensorBaseType);
#ifdef BUILD_METAL
INSTANTIATE_TYPED_TEST_SUITE_P(SoftMaxTestMetal, SoftMaxTestSuite, TensorMetalType);
#endif
#ifdef BUILD_CUDA
INSTANTIATE_TYPED_TEST_SUITE_P(SoftMaxTestCuda, SoftMaxTestSuite, TensorCudaType);
#endif
