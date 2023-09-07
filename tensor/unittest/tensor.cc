//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor_slowcpu.h>

#include "gtest/gtest.h"


template <typename _T, size_t _Rank, auto... _Args>
using Tensor = grid::TensorSlowCpu<_T, _Rank, _Args...>;

TEST(TensorSlowCPU, ParameterizedConstructor)
{
  // brace-initialization
  Tensor t11{ 11, 22, 33, 44, 55, 66 };
  Tensor t12{ { 11, 12 }, { 21, 22, 23 }, { 31, 32, 33, 34 } };

  EXPECT_EQ(t11.Rank(), 1);
  EXPECT_EQ(t12.Rank(), 2);

  EXPECT_EQ(t11.Dim(0), 6);
  EXPECT_EQ(t12.Dim(0), 3);

  int data[] = { 11, 22, 33, 44, 55, 66 };
  EXPECT_EQ(memcmp(t11.Data(), data, sizeof(data)), 0);

  // buffer allocation
  Tensor t21i(4UL, 1.2);
  Tensor t21u(5UL, grid::Uninitialized<double>{});
  Tensor t22i(5UL, 4UL, (char)3);
  Tensor t22u(7UL, 3UL, grid::Uninitialized<int>{});

  EXPECT_EQ(sizeof(*t22i.Data()), sizeof(char));

  EXPECT_EQ(t21i.Rank(), 1);
  EXPECT_EQ(t21u.Rank(), 1);
  EXPECT_EQ(t22i.Rank(), 2);
  EXPECT_EQ(t22u.Rank(), 2);

  EXPECT_EQ(t21i.Dim(0), 4);
  EXPECT_EQ(t21u.Dim(0), 5);
  EXPECT_EQ(t22i.Dim(0), 5);
  EXPECT_EQ(t22i.Dim(1), 4);
  EXPECT_EQ(t22u.Dim(0), 7);
  EXPECT_EQ(t22u.Dim(1), 3);

  double d21i[] = { 1.2, 1.2, 1.2, 1.2 };
  EXPECT_EQ(memcmp(t21i.Data(), d21i, sizeof(d21i)), 0);

  char d22i[] = { 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3 };
  EXPECT_EQ(memcmp(t22i.Data(), d22i, sizeof(d22i)), 0);

  Tensor t31{{4, 5, 7}, {4, 10, 10}, 3.3};
  EXPECT_EQ(t31.Rank(), 3);
  EXPECT_EQ(t31.Dim(0), 4);
  EXPECT_EQ(t31.Dim(1), 5);
  EXPECT_EQ(t31.Dim(2), 7);
  EXPECT_EQ(t31.Stride(0), 4);
  EXPECT_EQ(t31.Stride(1), 10);
  EXPECT_EQ(t31.Stride(2), 10);

  Tensor t32({1, 2, 3}, {4, 4, 4}, grid::Uninitialized<float>{});
  EXPECT_EQ(t31.Rank(), 3);
  EXPECT_EQ(t31.Dim(0), 4);
  EXPECT_EQ(t31.Dim(1), 5);
  EXPECT_EQ(t31.Dim(2), 7);
  EXPECT_EQ(t31.Stride(0), 4);
  EXPECT_EQ(t31.Stride(1), 10);
  EXPECT_EQ(t31.Stride(2), 10);

  size_t dim1[]{3UL, 4UL, 5UL};
  Tensor t41(dim1, dim1, 1.1);
  EXPECT_EQ(t41.Rank(), 3);

  size_t dim2[]{ 3, 4, 5 };
  size_t stride[] = { 4, 5, 6};
  Tensor t42(dim2, stride, grid::Uninitialized<double>{});
  EXPECT_EQ(t41.Rank(), 3);
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
  std::cout << res3 << std::endl;
  EXPECT_EQ(res3, v3);
}
