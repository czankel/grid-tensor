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

#include "../concepts.h"


namespace grid {

/// BinaryOperator<Operator> implements element-wise binary operations of two tensors.
/// The dimensions of the tensors must match following broadcasting rules.
/// The resulting rank is the maximum of the tensor ranks.
///
///  @tparm TOperator binary operator
template <typename TOperator>
class BinaryOperator
{
 private:

  // operation on a single element
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  0> dimensions,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    TOperator::eval(dest, src1, src2, 0);
  }

  // operation on a single dimension (unoptimized)
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  1> dimensions,
                  std::span<const ssize_t, 1> strides0,
                  std::span<const ssize_t, 1> strides1,
                  std::span<const ssize_t, 1> strides2) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      TOperator::eval(dest, src1, src2, 0);
      dest += strides0[0];
      src1 += strides1[0];
      src2 += strides2[0];
    }
  }

  template <size_t N, typename const_pointer, typename pointer> inline
  void eval(pointer dest, const_pointer src1, const_pointer src2,
            std::span<const size_t,  N> dimensions,
            std::span<const ssize_t, N> strides0,
            std::span<const ssize_t, N> strides1,
            std::span<const ssize_t, N> strides2) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      eval(dest, src1, src2,
           std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
           std::span<const ssize_t, N - 1>(strides0.begin() + 1, N - 1),
           std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1),
           std::span<const ssize_t, N - 1>(strides2.begin() + 1, N - 1));
      dest += strides0[0];
      src1 += strides1[0];
      src2 += strides2[0];
    }
  }

  //
  // Optimized (folded)
  //

  // operation on a single dimension (optimized)
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2, size_t dimensions) const
  {
    for (size_t i = 0; i < dimensions; i++)
      TOperator::eval(dest, src1, src2, i);
  }

  // operation on dim >= 2 (optimized)
  template <size_t N, typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                   std::span<const size_t,  N> dimensions,
                   std::span<const ssize_t, N> strides0,
                   std::span<const ssize_t, N> strides1,
                   std::span<const ssize_t, N> strides2,
                   size_t folded_dimensions) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      if constexpr (N > 1)
        eval(dest, src1, src2,
             std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
             std::span<const ssize_t, N - 1>(strides0.begin() + 1, N - 1),
             std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1),
             std::span<const ssize_t, N - 1>(strides2.begin() + 1, N - 1),
             folded_dimensions);
      else
        eval(dest, src1, src2, folded_dimensions);

      dest += strides0[0];
      src1 += strides1[0];
      src2 += strides2[0];
    }
  }

  // fold dimensions
  template <size_t N, typename const_pointer, typename pointer>
  inline void fold(pointer dest, const_pointer src1, const_pointer src2,
                   std::span<const size_t,  N> dimensions,
                   std::span<const ssize_t, N> strides0,
                   std::span<const ssize_t, N> strides1,
                   std::span<const ssize_t, N> strides2,
                   size_t folded_dimensions) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");

    if constexpr (N >= 2)
    {
      // "fold" one rank
      if (strides0[N - 2] - dimensions[N - 1] == 0 &&
          strides1[N - 2] - dimensions[N - 1] == 0 &&
          strides2[N - 2] - dimensions[N - 1] == 0)
      {
        fold(dest, src1, src2,
            std::span<const size_t,  N - 1>(dimensions.begin(), N - 1),
            std::span<const ssize_t, N - 1>(strides0.begin(), N - 1),
            std::span<const ssize_t, N - 1>(strides1.begin(), N - 1),
            std::span<const ssize_t, N - 1>(strides2.begin(), N - 1),
            folded_dimensions * dimensions[N - 2]);
        return;
      }
    }
    eval(dest, src1, src2,
         std::move(dimensions),
         std::move(strides0),
         std::move(strides1),
         std::move(strides2),
         folded_dimensions);
  }

 public:
  // FIXME: remove when all functions are converted to use ranges
  template <typename T, size_t TRank>
  void operator()(T* dest, const T* src1, const T* src2,
                  const std::array<size_t, TRank>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1,
                  const std::array<ssize_t, TRank>& strides2)
  {
    if constexpr (TRank > 2)
    {
      if (strides0[TRank - 2] - dimensions[TRank - 1] == 0 &&
          strides1[TRank - 2] - dimensions[TRank - 1] == 0 &&
          strides2[TRank - 2] - dimensions[TRank - 1] == 0)
        eval(dest, src1, src2,
             std::span<const size_t,  TRank - 1>(dimensions.begin(), TRank - 1),
             std::span<const ssize_t, TRank - 1>(strides0.begin(), TRank - 1),
             std::span<const ssize_t, TRank - 1>(strides1.begin(), TRank - 1),
             std::span<const ssize_t, TRank - 1>(strides2.begin(), TRank - 1),
             dimensions[TRank - 1]);
      return;
    }

    eval(dest, src1, src2,
         std::span<const size_t,  TRank>(dimensions),
         std::span<const ssize_t, TRank>(strides0),
         std::span<const ssize_t, TRank>(strides1),
         std::span<const ssize_t, TRank>(strides2));
  }

  template<std::ranges::input_range R1,
           std::ranges::input_range R2,
           std::weakly_incrementable O> //,
           //std::copy_constructible F>
  requires std::indirectly_copyable<std::ranges::iterator_t<R1>, O> &&
           std::indirectly_copyable<std::ranges::iterator_t<R2>, O>
  constexpr std::ranges::binary_transform_result<std::ranges::borrowed_iterator_t<R1>,
                                                 std::ranges::borrowed_iterator_t<R2>, O>
  operator()(R1&& r1, R2&& r2, O result/*, F op*/) const
  {
    constexpr size_t rank = O::rank;
    auto first1 = std::ranges::cbegin(r1);
    auto first2 = std::ranges::cbegin(r2);
    auto [strides1, strides2] = BroadcastStrides(first1, first2);
    auto& strides0 = result.Strides();
    auto& dimensions = result.Extents();

    // check if we can "fold" one rank
    if constexpr (rank > 2)
    {
      if (strides0[rank - 2] - dimensions[rank - 1] == 0 &&
          strides1[rank - 2] - dimensions[rank - 1] == 0 &&
          strides2[rank - 2] - dimensions[rank - 1] == 0)
      {
        fold(&*result, &*first1, &*first2,
             std::span<const size_t,  rank - 1>(dimensions.begin(), rank - 1),
             std::span<const ssize_t, rank - 1>(strides0.begin(), rank - 1),
             std::span<const ssize_t, rank - 1>(strides1.begin(), rank - 1),
             std::span<const ssize_t, rank - 1>(strides2.begin(), rank - 1),
             dimensions[rank - 1]);

        // FIXME: result needs to be at the end?
        return {std::move(std::ranges::end(r1)), std::move(std::ranges::end(r2)), std::move(result)};
      }
    }

    eval(&*result, &*first1, &*first2,
         std::span<const size_t,  rank>(dimensions.begin(), rank),
         std::span<const ssize_t, rank>(strides0.begin(), rank),
         std::span<const ssize_t, rank>(strides1.begin(), rank),
         std::span<const ssize_t, rank>(strides2.begin(), rank));

    // FIXME: result needs to be at the end?
    return {std::move(std::ranges::end(r1)), std::move(std::ranges::end(r2)), std::move(result)};
  }

};

//
// Operators
//

struct AddOperator
{
  // scalar X scalar
  template<typename T>
  static inline void eval(T* dest, const T* src1, const T* src2, const size_t i) { dest[i] = src1[i] + src2[i]; }
};

struct SubOperator
{
  // scalar X scalar
  template<typename T>
  static inline void eval(T* dest, const T* src1, const T* src2, const size_t i) { dest[i] = src1[i] - src2[i]; }
};

struct MulOperator
{
  // scalar X scalar
  template<typename T>
  static inline void eval(T* dest, const T* src1, const T* src2, const size_t i) { dest[i] = src1[i] * src2[i]; }
};

struct DivOperator
{
  // scalar X scalar
  template<typename T>
  static inline void eval(T* dest, const T* src1, const T* src2, const size_t i) { dest[i] = src1[i] / src2[i]; }
};


} // end of namespace grid

#endif // GRID_TENSOR_BASE_BINARY_H
