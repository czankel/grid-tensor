//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_CUDA_BINARY_H
#define GRID_TENSOR_CUDA_BINARY_H

#include <span>
#include <algorithm>
#include <ranges>

#include "device.h"
#include "../binary.h"
#include "../concepts.h"


namespace grid {

/// BinaryOperator<Operator> implements element-wise binary operations of two tensors.
/// The dimensions of the tensors must match following broadcasting rules.
/// The resulting rank is the maximum of the tensor ranks.
///
///  @tparm TOperator binary operator
template <template <typename> typename TOperator>
class BinaryOperator<TOperator<device::Cuda>>
{
  template <typename T>
  void eval(T*, const T*, const T*,
            std::span<const size_t, 0>,
            std::span<const ssize_t, 0>,
            std::span<const ssize_t, 0>,
            std::span<const ssize_t, 0>) const;
#if 0
  template <typename T, size_t 0>
  void eval(T*, const T*, const T*,
            std::span<const size_t, N>,
            std::span<const ssize_t, N>,
            std::span<const ssize_t, N>,
            std::span<const ssize_t, N>) const;
#endif
  // FIXME
#if 0
  // TODO: gcc doesn't like this constexpr, which would be use later as just Operator(args). Should it? See P0386R2 change: 9.2.3.2p3
  //static constexpr TOperator<device::Cuda> Operator;

  // operation on a single element
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  0> dimensions,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    TOperator<device::Cuda>()(dest, src1, src2, 0);
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
      TOperator<device::Cuda>()(dest, src1, src2, 0);
      dest += strides0[0];
      src1 += strides1[0];
      src2 += strides2[0];
    }
  }
#endif

  // operation on dim >=2 (unoptimized)
  template <size_t N, typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
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
#if 0
  // operation on a single dimension (optimized)
  template <typename const_pointer, typename pointer>
  void eval(pointer dest, const_pointer src1, const_pointer src2, size_t dimensions) const;
#if 0
  {
    for (size_t i = 0; i < dimensions; i++)
      TOperator<device::Cuda>()(dest, src1, src2, i);
  }
#endif
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
        ; //eval(dest, src1, src2, folded_dimensions);

      dest += strides0[0];
      src1 += strides1[0];
      src2 += strides2[0];
    }
  }
#endif
  // FIXME: use span.last instead of span.begin + 1 ... or subspan
  // FIXME: could this be done in the 'function' and initialized only once?


  // fully optimized contiguous tensors (only single dimension size)
  template <typename T> void eval(T*, const T*, const T*, size_t) const;

  // folded version with the folded dimensions first, followed by rank and pointers to dimensions and all strides
  template <typename T>
  void eval(T*, const T*, const T*, size_t, const size_t*, const ssize_t*, const ssize_t*, const ssize_t*) const;

  // unoptimized version with pointers to dimension and all strides
  template <typename T>
  void eval(T*, const T*, const T*, size_t, size_t, const size_t*, const ssize_t*, const ssize_t*, const ssize_t*) const;

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
        // FIXME: use subspan?
        fold(dest, src1, src2,
            std::span<const size_t,  N - 1>(dimensions.begin(), N - 1),
            std::span<const ssize_t, N - 1>(strides0.begin(), N - 1),
            std::span<const ssize_t, N - 1>(strides1.begin(), N - 1),
            std::span<const ssize_t, N - 1>(strides2.begin(), N - 1),
            folded_dimensions * dimensions[N - 2]);
        return;
      }
    }
    eval(dest, src1, src2, N, folded_dimensions, &*dimensions.begin(), &*strides0.begin(), &*strides1.begin(), &*strides2.begin());
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
      }
    }

    // FIXME: assert xx.size() == rank?
    eval(&*result, &*first1, &*first2, rank, &*dimensions.begin(), &*strides0.begin(), &*strides1.begin(), &*strides2.begin());
  }

};

} // end of namespace grid

#endif // GRID_TENSOR_CUDA_BINARY_H
