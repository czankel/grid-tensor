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
template <> class RmsNormOperator<device::Base>
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
  template<std::ranges::input_range R, std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<O>>
  void operator()(R&& r, O&& o) const
  {
    using tensor_type = std::remove_cvref_t<O>;
    using value_type = tensor_type::value_type;
    constexpr value_type eps = Eps<value_type>::default_value;

    auto first = std::ranges::cbegin(r);
    auto result = std::ranges::begin(o);
    auto& extents = result.Extents();

    auto [value, count] = SumSquare(&*first, std::span(extents), std::span(first.Strides()));

    value_type scale = 1.0f/sqrtf(value / count + eps);
    BinaryOperator<MulOperator<device::Base>>()(r, Tensor(scale), o);
  }
};

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_RMS_NORM_H
