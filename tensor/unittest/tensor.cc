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

TEST(TensorSlowCPU, Constructor)
{
  Tensor t21i(4, 1.2);
  Tensor t21u(5, grid::Uninitialized<double>{});
  Tensor t22i(5, 4, (char)3);
  Tensor t22u(7, 3, grid::Uninitialized<int>{});

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
}
