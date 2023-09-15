//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/slowcpu/tensor.h>
#include <grid/tensor/slowcpu/comparison.h>
#include <grid/tensor/slowcpu/rms_norm.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ElementsAre;

namespace {
template <typename _T> constexpr size_t size(size_t count) { return sizeof(_T) * count; }
}

template <typename _T, size_t _Rank, auto... _Args>
using Tensor = grid::TensorSlowCpu<_T, _Rank, _Args...>;

TEST(TensorSlowCPU, TensorNornRmsRank1)
{
  Tensor tensor{ 1.618, 2., 3.14, 5.382, -8.5, 13., -21., 34.77, 55. };
  Tensor result = grid::TensorRmsNorm(tensor);

  double scale = 23.47965240478515269729;
  Tensor expected{ 1.618 / scale, 2. / scale, 3.14 / scale, 5.382 / scale, -8.5 / scale,
                   13. / scale, -21. / scale, 34.77 / scale, 55. / scale };
  EXPECT_EQ(result, expected);
}

TEST(TensorSlowCPU, TensorNornRmsRank2)
{
  Tensor tensor{ { 1.618, 2., 3.14, 5.382, -8.5}, { 13., -21., 34.77, 55., 43.5} };
  auto op = grid::TensorRmsNorm(tensor);
  auto result = op();
}

