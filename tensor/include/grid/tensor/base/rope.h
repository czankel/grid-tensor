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

  eval()
  {
    // RoPE, rotate for each 'head'
    auto q = l.q_.Data();
    auto k = l.key_cache_.View(pos).Data();
    for (size_t i = 0; i < dim; i+=2)
    {
      float rot = (float) pos / powf(10000.0f, (float)(i % head_size) / (float)head_size);
      float fcr = cosf(rot);
      float fci = sinf(rot);

      float v0 = q[i];
      float v1 = q[i+1];
      q[i]   = v0 * fcr - v1 * fci;
      q[i+1] = v0 * fci + v1 * fcr;

      if (i < kv_dim)
      {
        float v0 = k[i];
        float v1 = k[i+1];
        k[i]   = v0 * fcr - v1 * fci;
        k[i+1] = v0 * fci + v1 * fcr;
      }
    }
  }

 public:
  template<std::ranges::input_range I,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    auto first_r = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);

    std::span strides_r(first_r.Strides());
    std::span strides_x(first_x.Strides());

    // FIXME: fold needed for rope?
    details::Fold([&](auto dimensions, bool contiguous) {
        if (contiguous)
          eval(&*first_r, &*first_x, dimensions,
               strides_r.template first<(dimensions.size() > 0) ? dimensions.size() - 1 : 0>(),
               strides_x);
        else
          eval(&*first_r, &*first_x, dimensions, strides_r, strides_x);
    }, std::span(first_r.Extents()), strides_r, strides_x);
  }

};

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_RMS_NORM_H
