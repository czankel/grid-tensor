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
  EXPECT_EQ(t21i.Rank(), 1);
  EXPECT_EQ(t21i.Dim(0), 4);
  double d21i[] = { 1.2, 1.2, 1.2, 1.2 };
  EXPECT_EQ(memcmp(t21i.Data(), d21i, sizeof(d21i)), 0);
}
