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
template <> class SiluFunction<device::Base>
{
 private:

  template <typename T>
  inline void
  Silu(T* d, const T* x,
       std::span<const size_t,  1> dimensions,
       std::span<const ssize_t, 1> strides_d,
       std::span<const ssize_t, 1> strides_x) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      *d = *x / (1.0f + exp(-*x));
      d += strides_d[0];
      x += strides_x[0];
    }
  }

  template <typename T, size_t _N>
  inline void
  Silu(T* d, const T* x,
       std::span<const size_t,  _N> dimensions,
       std::span<const ssize_t, _N> strides_d,
       std::span<const ssize_t, _N> strides_x) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      Silu(d, x,
           std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
           std::span<const ssize_t, _N - 1>(strides_d.begin() + 1, _N - 1),
           std::span<const ssize_t, _N - 1>(strides_x.begin() + 1, _N - 1));
      d += strides_d[0];
      x += strides_x[0];
    }
  }

 public:
  template<std::ranges::input_range I, std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);
    Silu(&*first_d, &*first_x, first_d.Extents(), first_x.Strides(), first_d.Strides());
  }
};

} // end of namespace grid

#endif // GRID_TENSOR_BASE_SILU_H
