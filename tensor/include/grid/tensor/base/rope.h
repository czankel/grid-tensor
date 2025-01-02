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
//#include "../tensor_operator.h"

namespace grid {
#if 0 // FIXME
namespace {
  template <typename> struct Eps {};
  template <> struct Eps<float>  { constexpr static float  default_value = 1e-5f; float  value; };
  template <> struct Eps<double> { constexpr static double default_value = 1e-5f; double value; };
}
#endif
// requires (std::is_floating_point_v<value_type> && rank > 0)
template <> struct RopeOperator<device::Base>
{
 private:

  template <typename T>
  inline void Eval(T* d, const T* x,
                   auto dimensions, auto strides_d, auto strides_x,
                   unsigned int pos) const

//  base_wavelength
            /*
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
  {
    static_assert(dimensions.size() != std::dynamic_extent, "dynamic_extent not allowed");
    //static_assert(dimensions.size() == 1, "rope only supports rank1 tensors");
#if 0
    int head_size = dimensions[sizeof(dimensions) - 1];
    printf("POS %d %ld\n", pos, dimensions[0]);

    for (size_t i = 0; i < head_size; i++)
      static ...xx[i] = powf(Base, static_cast<float>(i) / static_cast<float>head_size);

    for (size_t i = 0; i < head_size; i++)
    {
      float rot = powf(10000.0f, (float) i / (float) dimensions[1]);
      fcr[i] = cosf(rot);
      fci[i] = sinf(rot);
    }

    // FIXME: powf could be cached??
    for (size_t i = 0; i < dimensions[0] * dimensions[1]; i+=2)
    {
      // rot = pos * (1 / pow[i])
      float rot = (float) pos / powf(10000.0f, (float)(i % head_size) / (float)head_size);
      float fcr = cosf(rot);
      float fci = sinf(rot);

      float v0 = d[i];
      float v1 = d[i+1];
      d[i]   = v0 * fcr - v1 * fci;
      d[i+1] = v0 * fci + v1 * fcr;
    }
#endif
  }

 public:
  template<std::ranges::input_range I,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out, int pos) const
  {
    auto first_r = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);

    std::span strides_r(first_r.Strides());
    std::span strides_x(first_x.Strides());
    // FIXME: fold needed for rope? should result in a rank 1??
    // FIXME: should we support strides??
    details::Fold([&](auto dimensions, bool contiguous) {
        if (contiguous)
          Eval(&*first_r, &*first_x, dimensions,
               strides_r.template first<(dimensions.size() > 0) ? dimensions.size() - 1 : 0>(),
               strides_x, pos);
        else
          Eval(&*first_r, &*first_x, dimensions, strides_r, strides_x, pos);
    }, std::span(first_r.Extents()), strides_r, strides_x);
  }

};

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_ROPE_H
