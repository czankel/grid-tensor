//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_SILU_H
#define GRID_TENSOR_BASE_SILU_H

#include <limits>
#include <math.h>
#include <tuple>

#include "binary.h"

namespace grid {

/// Silu implements the SiLU activation function operator.
template <> class SiluOperator<device::Base>
{
 private:

  template <typename T>
  inline void
  Silu(T* dst, const T* src,
       std::span<const size_t,  1> dimensions,
       std::span<const ssize_t, 1> strides0,
       std::span<const ssize_t, 1> strides1) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      *dst = *src / (1.0f + exp(-*src));
      dst += strides0[0];
      src += strides1[0];
    }
  }

  template <typename T, size_t _N>
  inline void
  Silu(T* dst, const T* src,
       std::span<const size_t,  _N> dimensions,
       std::span<const ssize_t, _N> strides0,
       std::span<const ssize_t, _N> strides1) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      Silu(dst, src,
           std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
           std::span<const ssize_t, _N - 1>(strides0.begin() + 1, _N - 1),
           std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1));
      dst += strides0[0];
      src += strides1[0];
    }
  }

 public:
  template<std::ranges::input_range R, std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<O>>
  void operator()(R&& r, O&& o) const
  {
    auto first = std::ranges::cbegin(r);
    auto result = std::ranges::begin(o);
    Silu(&*result, &*first, result.Extents(), first.Strides(), result.Strides());
  }
};

} // end of namespace grid

#endif // GRID_TENSOR_BASE_SILU_H
