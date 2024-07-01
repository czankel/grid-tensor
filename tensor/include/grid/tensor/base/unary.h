//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_UNARY_H
#define GRID_TENSOR_BASE_UNARY_H

#include <span>
#include <algorithm>
#include <ranges>

#include "../concepts.h"
#include "../unary.h"
#include "../tensor_operator.h"

namespace grid {

/// UnaryOperator<Operator> implements element-wise unary operation on a tensors.
///
///  @tparm TOperator unary operator
template <template <typename> typename TOperator>
class UnaryOperator<TOperator, device::Base>
{
  template <typename T>
  inline void eval(T* dst, const T* src, auto dimensions, auto strides0, auto strides1) const
  {
    static_assert(dimensions.size() != std::dynamic_extent, "dynamic_extent not allowed");

    if constexpr (dimensions.size() == 0)
      dst[0] = TOperator<device::Base>()(src[0]);

    else if constexpr (dimensions.size() == 1  && strides0.size() == 0)
      for (size_t i = 0; i < dimensions[0]; i++)
        dst[i] = TOperator<device::Base>()(src[i]);

    else
      for (size_t i = 0; i < dimensions[0]; i++)
      {
        eval(dst, src,
             dimensions.template last<dimensions.size() - 1>(),
             strides0.template last<(strides0.size() > 0) ? strides0.size() - 1 : 0>(),
             strides1.template last<(strides1.size() > 0) ? strides1.size() - 1 : 0>());
        dst += strides0[0];
        src += strides1.size() > 0 ? strides1[0] : 0;
      }
  }

 public:
  template<std::ranges::input_range R,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<O>>
  void operator()(R&& r, O&& o) const
  {
    auto first =  std::ranges::cbegin(r);
    auto result = std::ranges::begin(o);

    std::span strides0(result.Strides());
    std::span strides1(first.Strides());

    details::Fold([&](auto dimensions, bool contiguous) {
        if (contiguous)
          eval(&*result, &*first, dimensions,
               strides0.template first<(dimensions.size() > 0) ? dimensions.size() - 1 : 0>(),
               strides1);
        else
          eval(&*result, &*first, dimensions, strides0, strides1);
    }, std::span(result.Extents()), strides0, strides1);
  }
};

//
// Operators
//

template <> struct CopyOperator<device::Base>
{
  template<typename T> inline T operator()(const T src) const { return src; }
};

template <> struct NegOperator<device::Base>
{
  template<typename T> inline T operator()(const T src) const { return -src; }
};


} // end of namespace grid

#endif // GRID_TENSOR_BASE_UNARY_H
