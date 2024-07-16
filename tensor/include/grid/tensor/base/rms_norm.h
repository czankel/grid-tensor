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
#include <iomanip>


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
  SumSquare(const T* src, const size_t dim, const ssize_t stride) const
  {
    T value{0};
    for (size_t i = 0; i < dim; i++, src += stride)
      value += *src * *src;
    return value;
  }

 public:
  template<std::ranges::input_range R, std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<O>>
  void operator()(R&& r, O&& o) const
  {
    using tensor_type = std::remove_cvref_t<O>;
    using value_type = tensor_type::value_type;
    constexpr value_type eps = Eps<value_type>::default_value;
    constexpr size_t rank = tensor_type::rank;

    auto first = std::ranges::cbegin(r);
    auto result = std::ranges::begin(o);
    auto& extents = result.Extents();

    size_t stride = first.Strides().back();
    size_t row_size = extents.back();
    size_t n_rows = std::accumulate(std::begin(extents), std::end(extents) - 1, 1, std::multiplies<size_t>());
    if (n_rows == 1)
    {
      auto sum = SumSquare(&*first, row_size, stride);
      value_type scale = sqrtf(sum / row_size + eps);
      BinaryOperator<DivOperator, device::Base>()(r, Tensor(scale), o);
    }
    else
    {
      // TODO could there be an alias issue?
      auto ptr = &*first;
      Tensor scale({n_rows}, Uninitialized<value_type>{});
      for (size_t row = 0; row < n_rows; row++)
      {
        auto sum = SumSquare(ptr, row_size, stride);
        scale.Data()[row] = sqrtf(sum / row_size + eps);
        ptr += first.Strides()[rank - 2];
      }
      BinaryOperator<DivOperator, device::Base>()(r, Tensor(scale), o);
    }
  }
};

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_RMS_NORM_H
