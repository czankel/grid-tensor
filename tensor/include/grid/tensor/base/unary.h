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
class UnaryOperator<TOperator<device::Base>>
{
  // static constexpr TOperator<device::Base> Operator();

  // operation on a single element
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dst, const_pointer src,
                   std::span<const size_t,  0> dimensions,
                   std::span<const ssize_t, 0>,
                   std::span<const ssize_t, 0>) const
  {
    dst[0] = TOperator<device::Base>()(src[0]);
  }

  // operation on a single dimension (contiguous)
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dst, const_pointer src,
                   std::span<const size_t,  1> dimensions,
                   std::span<const ssize_t, 0>,
                   std::span<const ssize_t, 0>) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
      dst[i] = TOperator<device::Base>()(src[i]);
  }

  // operation on a single dimension (non-contiguous)
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dst, const_pointer src,
                   std::span<const size_t,  1> dimensions,
                   std::span<const ssize_t, 1> strides0,
                   std::span<const ssize_t, 1> strides1) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      dst[0] = TOperator<device::Base>()(src[0]);
      dst += strides0[0];
      src += strides1[0];
    }
  }

  // operation on higher dimensions (unoptimized)
  template <size_t N, typename const_pointer, typename pointer> inline
  //template <size_t N> inline
  void eval(pointer dst, const_pointer src,
            std::span<const size_t,  N> dimensions,
            std::span<const ssize_t, N> strides0,
            std::span<const ssize_t, N> strides1) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      eval(dst, src,
           std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
           std::span<const ssize_t, N - 1>(strides0.begin() + 1, N - 1),
           std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1));
      dst += strides0[0];
      src += strides1[0];
    }
  }

 public:
  template<std::ranges::input_range R,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<O>>
  void operator()(R&& r, O&& o) const
  {
    using tensor_type = std::remove_cvref_t<O>;
    constexpr size_t rank = tensor_type::rank;
    auto first = std::ranges::cbegin(r);
    auto result = std::ranges::begin(o);
    auto& dimensions = result.Extents();
    auto& strides0 = result.Strides();
    auto& strides1 = first.Strides();

    if constexpr (rank > 1)
    {
      if (strides0[rank - 1] == 1 && strides1[rank - 1] == 1)
      {
        details::Fold(
            std::span<size_t, rank>(dimensions.begin(), rank),
            std::span<const ssize_t, rank - 1>(strides0.begin(), rank - 1),
            std::span<const ssize_t, rank - 1>(strides1.begin(), rank - 1),
            [=](auto f_dimensions, auto f_strides0, auto f_strides1) {
              eval(&*result, &*first, f_dimensions, f_strides0, f_strides1);
            });
        return;
      }
    }
    eval(&*result, &*first,
         std::span<const size_t, rank>(dimensions),
         std::span<const ssize_t, rank>(strides0),
         std::span<const ssize_t, rank>(strides1));
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
