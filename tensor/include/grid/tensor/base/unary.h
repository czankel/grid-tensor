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
#include "../tensor_operation.h"

namespace grid {

/// UnaryOperation<Operator> implements element-wise unary operation on a tensors.
///
///  @tparm TOperator unary operator
template <template <typename> typename TOperator>
class UnaryOperation<TOperator, device::Base>
{
  template <typename T>
  inline void eval(T* d, const T* x, auto dimensions, auto strides_d, auto strides_x) const
  {
    static_assert(dimensions.size() != std::dynamic_extent, "dynamic_extent not allowed");

    if constexpr (dimensions.size() == 0)
      d[0] = TOperator<device::Base>()(x[0]);

    else if constexpr (dimensions.size() == 1  && strides_d.size() == 0)
      for (size_t i = 0; i < dimensions[0]; i++)
        d[i] = TOperator<device::Base>()(x[i]);

    else
      for (size_t i = 0; i < dimensions[0]; i++)
      {
        eval(d, x,
             dimensions.template last<dimensions.size() - 1>(),
             strides_d.template last<(strides_d.size() > 0) ? strides_d.size() - 1 : 0>(),
             strides_x.template last<(strides_x.size() > 0) ? strides_x.size() - 1 : 0>());
        d += strides_d[0];
        x += strides_x.size() > 0 ? strides_x[0] : 0;
      }
  }

 public:
  template<std::ranges::input_range I,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);

    std::span strides_d(first_d.Strides());
    std::span strides_x(first_x.Strides());

    Fold([&](auto dimensions, bool contiguous) {
        if (contiguous)
          eval(&*first_d, &*first_x, dimensions,
               strides_d.template first<(dimensions.size() > 0) ? dimensions.size() - 1 : 0>(),
               strides_x);
        else
          eval(&*first_d, &*first_x, dimensions, strides_d, strides_x);
    }, std::span(first_d.Extents()), strides_d, strides_x);
  }
};

//
// Elementary Unary Operators
//

template <> struct CopyOperator<device::Base>
{
  template<typename T> inline T operator()(const T x) const { return x; }
};

template <> struct NegOperator<device::Base>
{
  template<typename T> inline T operator()(const T x) const { return -x; }
};

//
// Unary Operations
//

template <> struct SiluFunction<device::Base>
{
  template<typename T> inline T operator()(const T x) const { return x * (T{1} + exp(-x)); }
};


} // end of namespace grid

#endif // GRID_TENSOR_BASE_UNARY_H
