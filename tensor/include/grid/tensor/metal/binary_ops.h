//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_METAL_BINARY_OPS_H
#define GRID_TENSOR_METAL_BINARY_OPS_H

#include "device.h"

namespace grid {

template <typename TOperator>
class BinaryOperator<device::Metal, TOperator>
{

 public:
  template <typename T, size_t TRank>
  void operator()(T* dst, const T* src1, const T* src2,
                  const std::array<size_t,  TRank>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1,
                  const std::array<ssize_t, TRank>& strides2)
  {
    if constexpr (TRank > 2)
    {
      if (strides0[TRank - 2] - dimensions[TRank - 1] == 0 &&
          strides1[TRank - 2] - dimensions[TRank - 1] == 0 &&
          strides2[TRank - 2] - dimensions[TRank - 1] == 0)
        eval(dst, src1, src2,
             dimensions[TRank - 1],
             std::span<const size_t,  TRank - 1>(dimensions.begin(), TRank - 1),
             std::span<const ssize_t, TRank - 1>(strides0.begin(), TRank - 1),
             std::span<const ssize_t, TRank - 1>(strides1.begin(), TRank - 1),
             std::span<const ssize_t, TRank - 1>(strides2.begin(), TRank - 1));
      return;
    }

    eval(dst, src1, src2,
         std::span<const size_t,  TRank>(dimensions),
         std::span<const ssize_t, TRank>(strides0),
         std::span<const ssize_t, TRank>(strides1),
         std::span<const ssize_t, TRank>(strides2));
  }
};


} // end of namespace grid

#endif  // GRID_TENSOR_METAL_BINARY_OPS_H

