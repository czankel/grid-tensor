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

TEST(TensorSlowCPU, BraceInitializationRank1Integer)
{
  Tensor tensor1{ 11, 22, 33, 44, 55, 66 };

  EXPECT_TRUE(grid::is_tensor_v<decltype(tensor1)>);
  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(6));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<int>(1)));

  int data[] = { 11, 22, 33, 44, 55, 66 };
  EXPECT_EQ(memcmp(tensor1.Data(), data, sizeof(data)), 0);
}

TEST(TensorSlowCPU, BraceInitializationRank2Integer)
{
  Tensor tensor1{ { 11, 12 }, { 21, 22, 23 }, { 31, 32, 33, 34 } };

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(3, 4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<int>(4), size<int>(1)));

  const int* data = reinterpret_cast<const int*>(tensor1.Data());
  EXPECT_EQ(data[0], 11);
  EXPECT_EQ(data[4], 21);
  EXPECT_EQ(data[8], 31);
  EXPECT_EQ(data[9], 32);
}

TEST(TensorSlowCPU, AllocInitializedRank1Double)
{
  Tensor tensor1(4UL, 1.2);

  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(1)));

  double verify[] = { 1.2, 1.2, 1.2, 1.2 };
  EXPECT_EQ(memcmp(tensor1.Data(), verify, sizeof(verify)), 0);
}

TEST(TensorSlowCPU, AllocUninitializedRank1Double)
{
  Tensor tensor1(5UL, grid::Uninitialized<double>{});
  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(5));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(1)));
}

TEST(TensorSlowCPU, AllocInitializedRank2Char)
{
  Tensor tensor1(5UL, 4UL, (char)'3');

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(5, 4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<char>(4), size<char>(1)));

  char verify[] = { '3', '3', '3', '3', '3',
                  '3', '3', '3', '3', '3',
                  '3', '3', '3', '3', '3',
                  '3', '3', '3', '3', '3' };
  EXPECT_EQ(memcmp(tensor1.Data(), verify, sizeof(verify)), 0);
}

TEST(TensorSlowCPU, AllocUninitializedRank2Double)
{
  Tensor tensor1(7UL, 3UL, grid::Uninitialized<int>{});

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(7, 3));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<int>(3), size<int>(1)));
}

TEST(TensorSlowCPU, AllocUninitializedPattedRank3Double)
{
  Tensor tensor1({3, 2, 1}, {size<double>(2 * 2 * 4), size<double>(2 * 2), size<double>(2)}, grid::Uninitialized<double>{});
  EXPECT_EQ(tensor1.Rank(), 3);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(3, 2, 1));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(2 * 2 * 4), size<double>(2 * 2), size<double>(2)));
}


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
