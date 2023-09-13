//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor_slowcpu.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ElementsAre;

namespace {
template <typename _T> constexpr size_t size(size_t count) { return sizeof(_T) * count; }
}

template <typename _T, size_t _Rank, auto... _Args>
using Tensor = grid::TensorSlowCpu<_T, _Rank, _Args...>;

TEST(TensorSlowCPU, TensorAdd)
{
  Tensor t11{ 11, 22, 33, 44, 55, 66 };
  Tensor t12{ 89, 78, 67, 56, 45, 34 };
  int v1[] = { 100, 100, 100, 100, 100, 100 };

  auto op1a = grid::TensorAdd(t11, t12);
  auto res1a = op1a();
  EXPECT_EQ(memcmp(res1a.Data(), v1, sizeof(v1)), 0);

  auto&& op1b = t12 + t11;
  auto res1b = op1b();
  EXPECT_EQ(memcmp(res1b.Data(), v1, sizeof(v1)), 0);

  Tensor t21(4UL, 5UL, 1.1);
  Tensor t22(4UL, 5UL, 2.4);

  auto op2 = std::move(t21) + std::move(t22);
  auto res2 = op2();
  Tensor v2{
    { 3.5, 3.5, 3.5, 3.5, 3.5 },
    { 3.5, 3.5, 3.5, 3.5, 3.5 },
    { 3.5, 3.5, 3.5, 3.5, 3.5 },
    { 3.5, 3.5, 3.5, 3.5, 3.5 }
  };
  EXPECT_EQ(res2, v2);
}

TEST(TensorSlowCPU, TensorAddAdd)
{
  Tensor t31(4UL, 3UL, 2.1);
  Tensor t32(4UL, 3UL, 1.3);
  Tensor t33(4UL, 3UL, 2.2);
  Tensor v3{ { 5.6, 5.6, 5.6 }, { 5.6, 5.6, 5.6 }, { 5.6, 5.6, 5.6 }, { 5.6, 5.6, 5.6 }};

  auto&& op31 = t31 + t32 + t33;
  auto res3 = op31();
  EXPECT_EQ(res3, v3);
}
