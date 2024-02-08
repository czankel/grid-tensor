//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_SOFTMAX_H
#define GRID_TENSOR_BASE_SOFTMAX_H

#include <limits>
#include <math.h>
#include <tuple>

#include "binary_ops.h"

namespace grid {

/// SoftMaxOperator implements the softmax operator.
class SoftMaxOperator
{
 private:

  template <typename T>
  inline auto
  Max(const T* src,
      std::span<const size_t,  1> dimensions,
      std::span<const ssize_t, 1> strides) const
  {
    T max{std::numeric_limits<T>::lowest()};

    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
      max = std::max(max, *src);

    return max;
  }

  template <typename T, size_t _N>
  inline auto
  Max(const T* src,
      std::span<const size_t,  _N> dimensions,
      std::span<const ssize_t, _N> strides) const
  {
    T max{std::numeric_limits<T>::lowest()};
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");

    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
    {
      max = std::max(max, Max(src,
                              std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
                              std::span<const ssize_t, _N - 1>(strides.begin() + 1, _N - 1)));
    }
    return max;
  }

  template <typename T>
  inline auto
  SumExp(T* dst,
         const T* src,
         T max,
         std::span<const size_t,  1> dimensions,
         std::span<const ssize_t, 1> strides) const
  {
    T sum{0};
    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
    {
      dst[i] = exp(*src - max);
      sum += dst[i];
    }
    return sum;
  }

  // FIXME: add strides for dst
  template <typename T, size_t _N>
  inline auto
  SumExp(T* dst,
         const T* src,
         T max,
         std::span<const size_t,  _N> dimensions,
         std::span<const ssize_t, _N> strides) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");

    T sum{0};
    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
    {
      sum += SumExp(dst, src, max,
                    std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
                    std::span<const ssize_t, _N - 1>(strides.begin() + 1, _N - 1));
    }
    return sum;
  }

 public:

  /// operator()() executes the operation and returns a tensor.
  // TODO: make eps a parameter, see also definition in rms_norm
  template <typename T, size_t TRank>
  auto operator()(T* dst, const T* src,
                  const std::array<size_t,  TRank>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1)
  {
    constexpr T eps = std::numeric_limits<T>::epsilon();

    auto max = Max(src, std::span(dimensions), std::span(strides1));
    auto sum = SumExp(dst, src, max, std::span(dimensions), std::span(strides1));

    T scale = static_cast<T>(1)/(sum + eps);

    auto strides2 = std::array<ssize_t, TRank>{0};
    BinaryOperator<MulOperator>{}(dst, dst, &scale, dimensions, strides0, strides0, strides2);
  }
};

} // end of namespace grid

#endif // GRID_TENSOR_BASE_SOFTMAX_H
