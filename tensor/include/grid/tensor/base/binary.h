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
class BinaryOperator<TOperator<device::Base>>
{
  // TODO: gcc doesn't like this constexpr, which would be use later as just Operator(args).
  // Should it? See P0386R2 change: 9.2.3.2p3
  // static constexpr TOperator<device::Base> Operator;

  // operation on a single element
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  0> dimensions,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    dest[0] = TOperator<device::Base>()(src1[0], src2[0]);
  }

  // operation on a single dimension (contiguous)
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  1> dimensions,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
      dest[i] = TOperator<device::Base>()(src1[i], src2[i]);
  }

  // operation on a single dimension (non-contiguous)
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  1> dimensions,
                  std::span<const ssize_t, 1> strides0,
                  std::span<const ssize_t, 1> strides1,
                  std::span<const ssize_t, 1> strides2) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      dest[0] = TOperator<device::Base>()(src1[0], src2[0]);
      dest += strides0[0];
      src1 += strides1[0];
      src2 += strides2[0];
    }
  }

  // operation on dim >=2 (optimized N != M or unoptimized N == M)
  template <size_t N, size_t M, typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                   std::span<const size_t,  N> dimensions,
                   std::span<const ssize_t, M> strides0,
                   std::span<const ssize_t, M> strides1,
                   std::span<const ssize_t, M> strides2) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      eval(dest, src1, src2,
           dimensions.template last<N - 1>(),
           strides0.template last<M - 1>(),
           strides1.template last<M - 1>(),
           strides2.template last<M - 1>());
      dest += strides0[0];
      src1 += strides1[0];
      src2 += strides2[0];
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
    using tensor_type = std::remove_cvref_t<O>;
    constexpr size_t rank = tensor_type::rank;
    auto first1 = std::ranges::cbegin(r1);
    auto first2 = std::ranges::cbegin(r2);
    auto result = std::ranges::begin(o);

    auto& dimensions = result.Extents();
    auto& strides0 = result.Strides();
    auto [strides1, strides2] = BroadcastStrides(std::span(first1.Strides()), std::span(first2.Strides()));

    if constexpr (rank > 1)
    {
      if (strides0[rank - 1] == 1 && strides1[rank - 1] == 1 && strides2[rank - 1] == 1)
      {
        details::Fold(
            std::span(dimensions),
            std::span<const ssize_t, rank - 1>(strides0.begin(), rank - 1),
            std::span<const ssize_t, rank - 1>(strides1.begin(), rank - 1),
            std::span<const ssize_t, rank - 1>(strides2.begin(), rank - 1),
            [=](auto f_dimensions, auto f_strides0, auto f_strides1, auto f_strides2) {
              eval(&*result, &*first1, &*first2, f_dimensions, f_strides0, f_strides1, f_strides2);
            });
        return;
      }
    }
    eval(&*result, &*first1, &*first2,
         std::span<const size_t, rank>(dimensions.begin(), rank),
         std::span<const ssize_t, rank>(strides0.begin(), rank),
         std::span<const ssize_t, rank>(strides1.begin(), rank),
         std::span<const ssize_t, rank>(strides2.begin(), rank));
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
