//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_ROPE_H
#define GRID_TENSOR_BASE_ROPE_H

#include <math.h>
#include <tuple>
#include <iomanip>

#include "../function.h"

namespace grid {

// requires (std::is_floating_point_v<value_type> && rank > 0)
template <> struct RopeOperator<device::Base>
{
 private:

            /*
  template <typename T>
  inline void Eval(T* d, const T* x,
                   auto dimensions, auto strides_d, auto strides_x,
                   unsigned int pos) const
//  base_wavelength

            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);
            */
 public:
  template<std::ranges::input_range I,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out, int pos) const
  {
    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);

    std::span strides_d(first_d.Strides());
    std::span strides_x(first_x.Strides());

    std::span dimensions(first_d.Extents());

    static_assert(dimensions.size() != std::dynamic_extent, "dynamic_extent not allowed");
    static_assert(dimensions.size() <= 2, "rope only supports max rank-2 tensors");

    size_t n_rows = std::accumulate(std::begin(dimensions),
                                    std::end(dimensions) - 1, 1,
                                    std::multiplies<size_t>());
    size_t n_cols = dimensions.back();

    auto x = &*first_x;
    auto d = &*first_d;

    if ((n_cols & 1) != 0)
      throw std::runtime_error("rope dimensions must be multiple of two");

    size_t i = 0;
    for (size_t r = 0; r < n_rows; r++)
    {
      for (size_t c = 0; c < n_cols/2; c++, i += 2)
      {
        float rot = (float) pos / powf(10000.0f, (float)c / (float)n_cols);
        float fcr = cosf(rot);
        float fci = sinf(rot);

        // FIXME: this is an usafe access?? can this be a span/vec/... that has a check?
        float v0 = x[i];
        float v1 = x[i+1];
        d[i]   = v0 * fcr - v1 * fci;
        d[i+1] = v0 * fci + v1 * fcr;
      }
    }
  }
};

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_ROPE_H
