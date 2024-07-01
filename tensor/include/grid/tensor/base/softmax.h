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

#include "binary.h"

namespace grid {

/// SoftMaxOperator implements the softmax operator.
template <> class SoftMaxOperator<device::Base>
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
  SumExp(T* dst, const T* src, T max,
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
  SumExp(T* dst, const T* src, T max,
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
  template<std::ranges::input_range R, std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<O>>
  void operator()(R&& r, O&& o) const
  {
    using tensor_type = std::remove_cvref_t<O>;
    using value_type = tensor_type::value_type;
    constexpr value_type eps = std::numeric_limits<value_type>::epsilon();

    auto first = std::ranges::cbegin(r);
    auto result = std::ranges::begin(o);
    auto& extents = result.Extents();
    auto max = Max(&*first, std::span(extents), std::span(result.Strides()));
    auto sum = SumExp(&*result, &*first, max, std::span(extents), std::span(first.Strides()));

    value_type scale = static_cast<value_type>(1)/(sum + eps);
    BinaryOperator<MulOperator, device::Base>()(o, Tensor(scale), o);
  }
};

} // end of namespace grid

#endif // GRID_TENSOR_BASE_SOFTMAX_H
