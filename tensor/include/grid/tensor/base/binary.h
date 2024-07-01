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
#include "../tensor_operator.h"

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
  inline void eval(T* dest, const T* src1, const T* src2, auto dimensions,
                   auto strides0, auto strides1, auto strides2) const
  {
    static_assert(dimensions.size() != std::dynamic_extent, "dynamic_extent not allowed");

    if constexpr (dimensions.size() == 0)
      dest[0] = TOperator<device::Base>()(src1[0], src2[0]);

    else if constexpr (dimensions.size() == 1  && strides0.size() == 0)
      for (size_t i = 0; i < dimensions[0]; i++)
        dest[i] = TOperator<device::Base>()(src1[i], src2[i]);

    else
      for (size_t i = 0; i < dimensions[0]; i++)
      {
        eval(dest, src1, src2,
             dimensions.template last<dimensions.size() - 1>(),
             strides0.template last<(strides0.size() > 0) ? strides0.size() - 1 : 0>(),
             strides1.template last<(strides1.size() > 0) ? strides1.size() - 1 : 0>(),
             strides2.template last<(strides2.size() > 0) ? strides2.size() - 1 : 0>());
        dest += strides0[0];
        src1 += strides1.size() > 0 ? strides1[0] : 0;
        src2 += strides2.size() > 0 ? strides2[0] : 0;
      }
  }

 public:
  template<std::ranges::input_range R1,
           std::ranges::input_range R2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<O>> &&
           std::indirectly_copyable<std::ranges::iterator_t<R2>, std::ranges::iterator_t<O>>
  void operator()(R1&& r1, R2&& r2, O&& o) const
  {
    auto first1 = std::ranges::cbegin(r1);
    auto first2 = std::ranges::cbegin(r2);
    auto result = std::ranges::begin(o);

    std::span strides0(result.Strides());
    std::span strides1(first1.Strides());
    std::span strides2(first2.Strides());

    details::Fold([&](auto dimensions, bool contiguous) {
        if (contiguous)
          eval(&*result, &*first1, &*first2, dimensions,
               strides0.template first<(dimensions.size() > 0) ? dimensions.size() - 1 : 0>(),
               strides1, strides2);
        else
          eval(&*result, &*first1, &*first2, dimensions, strides0, strides1, strides2);
    }, std::span(result.Extents()), strides0, strides1, strides2);
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
