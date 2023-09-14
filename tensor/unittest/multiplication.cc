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
#include <grid/tensor/slowcpu/multiplication.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ElementsAre;

namespace {
template <typename _T> constexpr size_t size(size_t count) { return sizeof(_T) * count; }
}

template <typename _T, size_t _Rank, auto... _Args>
using Tensor = grid::TensorSlowCpu<_T, _Rank, _Args...>;

TEST(TensorSlowCPU, TensorVecDot)
{
  // dot -> 14+33+65 = 112
  Tensor tensor1{   2,   3,   5 };
  Tensor tensor2{   7,  11,  13 };

  auto op = grid::TensorMul(tensor1, tensor2);
  auto res = op();
  EXPECT_EQ(res, Tensor{112});
}


TEST(TensorSlowCPU, TensorMatMul)
{
  Tensor tensor1{ { 3, 6, 9 },
                  { 2, 8, 4 }};
  Tensor tensor2{ { 1, 8 },
                  { 5, 3 },
                  { 7, 4 } };
  Tensor expected{{ 3 * 1 + 6 * 5 + 9 * 7, 3 * 8 + 6 * 3 + 9 * 4 },
                  { 2 * 1 + 8 * 5 + 4 * 7, 2 * 8 + 8 * 3 + 4 * 4 }};

  auto op = grid::TensorMul(tensor1, tensor2);
  auto res = op();
  EXPECT_EQ(res, expected);
}
