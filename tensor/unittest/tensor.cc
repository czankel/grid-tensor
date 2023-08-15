//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor_slowcpu.h>

#include "gtest/gtest.h"

template <size_t _Rank, typename _T, auto... _Args>
using Tensor = grid::TensorSlowCpu<_Rank, _T, _Args...>;

TEST(TensorSlowCPU, IniitalizerConstructor)
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
  Tensor t21i(4U, 1.2);
  Tensor t21u(5U, grid::Uninitialized<double>{});
  Tensor t22i(5U, 4U, (char)3);
  Tensor t22u(7U, 3U, grid::Uninitialized<int>{});

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
}
