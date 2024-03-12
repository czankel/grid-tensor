//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_RMS_NORM_H
#define GRID_TENSOR_BASE_RMS_NORM_H

#include <math.h>
#include <tuple>

#include "binary.h"

namespace grid {

namespace {
  template <typename> struct Eps {};
  template <> struct Eps<float>  { constexpr static float  default_value = 1e-5f; float  value; };
  template <> struct Eps<double> { constexpr static double default_value = 1e-5f; double value; };
}

// requires (std::is_floating_point_v<value_type> && rank > 0)
class RmsNormOperator
{
 private:
  template <typename T>
  inline auto
  SumSquare(const T* src,
            std::span<const size_t,  1> dimensions,
            std::span<const ssize_t, 1> strides) const
  {
    T value{0};
    size_t count = dimensions[0];
    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
      value += *src * *src;
    return std::tuple{value, count};
  }

  template <typename T, size_t N>
  inline auto
  SumSquare(const T* src,
            std::span<const size_t,  N> dimensions,
            std::span<const ssize_t, N> strides) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
    T value{0};
    size_t count = 0;
    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
    {
      auto [s, c] = SumSquare(src,
                              std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
                              std::span<const ssize_t, N - 1>(strides.begin() + 1, N - 1));
      value += s;
      count += c;
    }
    return std::tuple{value, count};
  }

 public:

  /// operator()() executes and returns a tensor with the RMS norm of the stored vector.
  template <typename T, size_t TRank>
  void operator()(T* dst, const T* src,
                  const std::array<size_t,  TRank>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1,
                  T eps = Eps<T>::default_value)
  {
    auto [value, count] = SumSquare(src, std::span(dimensions), std::span(strides1));
    T scale = 1.0f/sqrtf(value / count + eps);
    auto strides2 = std::array<ssize_t, TRank>{0};
    BinaryOperator<MulOperator>{}(dst, src, &scale, dimensions, strides0, strides1, strides2);
  }
};

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_RMS_NORM_H
