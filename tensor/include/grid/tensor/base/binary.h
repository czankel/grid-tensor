//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_BINARY_H
#define GRID_TENSOR_BASE_BINARY_H

#include <span>
#include <algorithm>
#include <ranges>

#include "../binary.h"
#include "../concepts.h"
#include "../tensor_operation.h"

namespace grid {

/// BinaryOperator<Operator> implements element-wise binary operations of two tensors.
/// The dimensions of the tensors must match following broadcasting rules.
/// The resulting rank is the maximum of the tensor ranks.
///
///  @tparm TOperator binary operator
template <template <typename> typename TOperator>
class BinaryOperator<TOperator, device::Base>
{
  // TODO: gcc doesn't like this constexpr, which would be use later as just Operator(args).
  // Should it? See P0386R2 change: 9.2.3.2p3
  // static constexpr TOperator<device::Base> Operator;

  template <typename T>
  inline void eval(T* d, const T* x, const T* y,
                   auto dimensions, auto strides_d, auto strides_x, auto strides_y) const
  {
    static_assert(dimensions.size() != std::dynamic_extent, "dynamic_extent not allowed");

    if constexpr (dimensions.size() == 0)
      d[0] = TOperator<device::Base>()(x[0], y[0]);

    else if constexpr (dimensions.size() == 1  && strides_d.size() == 0)
      for (size_t i = 0; i < dimensions[0]; i++)
        d[i] = TOperator<device::Base>()(x[i], y[i]);

    else
      for (size_t i = 0; i < dimensions[0]; i++)
      {
        eval(d, x, y,
             dimensions.template last<dimensions.size() - 1>(),
             strides_d.template last<(strides_d.size() > 0) ? strides_d.size() - 1 : 0>(),
             strides_x.template last<(strides_x.size() > 0) ? strides_x.size() - 1 : 0>(),
             strides_y.template last<(strides_y.size() > 0) ? strides_y.size() - 1 : 0>());
        d += strides_d[0];
        x += strides_x.size() > 0 ? strides_x[0] : 0;
        y += strides_y.size() > 0 ? strides_y[0] : 0;
      }
  }

 public:
  template<std::ranges::input_range I1,
           std::ranges::input_range I2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I1>, std::ranges::iterator_t<O>> &&
           std::indirectly_copyable<std::ranges::iterator_t<I2>, std::ranges::iterator_t<O>>
  void operator()(I1&& in1, I2&& in2, O&& out) const
  {
    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in1);
    auto first_y = std::ranges::cbegin(in2);

    std::span strides_d(first_d.Strides());
    std::span strides_x(first_x.Strides());
    std::span strides_y(first_y.Strides());

    details::Fold([&](auto dimensions, bool contiguous) {
        if (contiguous)
          eval(&*first_d, &*first_x, &*first_y, dimensions,
               strides_d.template first<(dimensions.size() > 0) ? dimensions.size() - 1 : 0>(),
               strides_x, strides_y);
        else
          eval(&*first_d, &*first_x, &*first_y, dimensions, strides_d, strides_x, strides_y);
    }, std::span(first_d.Extents()), strides_d, strides_x, strides_y);
  }
};

//
// Operators
//

template<> struct AddOperator<device::Base>
{
  template<typename T> inline T operator()(T a, T b) const { return a + b; }
};

template<> struct SubOperator<device::Base>
{
  template<typename T> inline T operator()(T a, T b) const { return a - b; }
};

template<> struct MulOperator<device::Base>
{
  template<typename T> inline T operator()(T a, T b) const { return a * b; }
};

template<> struct DivOperator<device::Base>
{
  template<typename T> inline T operator()(T a, T b) const { return a / b; }
};


} // end of namespace grid

#endif // GRID_TENSOR_BASE_BINARY_H
