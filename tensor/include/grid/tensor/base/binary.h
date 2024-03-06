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
#include "../device.h"

namespace grid {

/// BinaryOperator<Operator> implements element-wise binary operations of two tensors.
/// The dimensions of the tensors must match following broadcasting rules.
/// The resulting rank is the maximum of the tensor ranks.
///
///  @tparm TOperator binary operator
template <template <typename> typename TOperator, typename T>
class BinaryOperator<TOperator<device::Base>, T>
{
  static constexpr TOperator<device::Base> Operator();

  // operation on a single element
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  0> dimensions,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    Operator(dest, src1, src2, 0);
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
      TOperator<device::Base>{}(dest, src1, src2, 0);
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
  // optimized versions for contiguous tensors
  //

  template <size_t N, typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                   size_t dimensions0,
                   std::span<const size_t,  0>,
                   std::span<const ssize_t, 0>,
                   std::span<const ssize_t, 0>,
                   std::span<const ssize_t, 0>)
  {
    for (size_t i = 0; i < dimensions0; i++)
      TOperator<device::Base>::eval(dest, src1, src2, i);
  }


  // optimized for contiguous tensors
  template <size_t N, typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                   size_t dimensions0,
                   std::span<const size_t,  N> dimensions,
                   std::span<const ssize_t, N> strides0,
                   std::span<const ssize_t, N> strides1,
                   std::span<const ssize_t, N> strides2) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
    // "fold" one rank
    if (strides0[N - 2] - dimensions[N - 1] == 0 &&
        strides1[N - 2] - dimensions[N - 1] == 0 &&
        strides2[N - 2] - dimensions[N - 1] == 0)
    {
      eval(dest, src1, src2,
           dimensions[N - 2] * dimensions[N - 1],
           std::span<const size_t,  N - 1>(dimensions.begin(), N - 1),
           std::span<const ssize_t, N - 1>(strides0.begin(), N - 1),
           std::span<const ssize_t, N - 1>(strides1.begin(), N - 1),
           std::span<const ssize_t, N - 1>(strides2.begin(), N - 1));
    }
    else
    {
      for (size_t i = 0; i < dimensions0; i++)
      {
        eval(dest, src1, src2,
             dimensions[N - 1],
             std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
             std::span<const ssize_t, N - 1>(strides0.begin() + 1, N - 1),
             std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1),
             std::span<const ssize_t, N - 1>(strides2.begin() + 1, N - 1));
        dest += strides0[0];
        src1 += strides1[0];
        src2 += strides2[0];
      }
    }
  }

 public:
  template <typename TTensor0, typename TTensor1, typename TTensor2, size_t TRank>
  void operator()(TTensor0& result, const TTensor1& tensor1, const TTensor2& tensor2,
                  const std::array<size_t,  TRank>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1,
                  const std::array<ssize_t, TRank>& strides2) const
  {
    typename TTensor0::pointer dst = result.Data();
    typename TTensor1::const_pointer src1 = tensor1.Data();
    typename TTensor2::const_pointer src2 = tensor2.Data();
    if constexpr (TRank > 2)
    {
      if (strides0[TRank - 2] - dimensions[TRank - 1] == 0 &&
          strides1[TRank - 2] - dimensions[TRank - 1] == 0 &&
          strides2[TRank - 2] - dimensions[TRank - 1] == 0)
        eval(dst, src1, src2,
             dimensions[TRank - 1],
             std::span<const size_t,  TRank - 1>(dimensions.begin(), TRank - 1),
             std::span<const ssize_t, TRank - 1>(strides0.begin(), TRank - 1),
             std::span<const ssize_t, TRank - 1>(strides1.begin(), TRank - 1),
             std::span<const ssize_t, TRank - 1>(strides2.begin(), TRank - 1));
      return;
    }

    eval(dst, src1, src2,
         std::span<const size_t,  TRank>(dimensions),
         std::span<const ssize_t, TRank>(strides0),
         std::span<const ssize_t, TRank>(strides1),
         std::span<const ssize_t, TRank>(strides2));
  }
};


//
// Operators
//

template<> struct AddOperator<device::Base>
{
  template<typename T>
  inline void operator()(T* dest, const T* src1, const T* src2, const size_t i) { dest[i] = src1[i] + src2[i]; }
};

template<> struct SubOperator<device::Base>
{
  template<typename T>
  inline void operator()(T* dest, const T* src1, const T* src2, const size_t i) { dest[i] = src1[i] - src2[i]; }
};

template<> struct MulOperator<device::Base>
{
  template<typename T>
  inline void operator()(T* dest, const T* src1, const T* src2, const size_t i) { dest[i] = src1[i] * src2[i]; }
};

template<> struct DivOperator<device::Base>
{
  template<typename T>
  inline void operator()(T* dest, const T* src1, const T* src2, const size_t i) { dest[i] = src1[i] / src2[i]; }
};


} // end of namespace grid

#endif // GRID_TENSOR_BASE_BINARY_H
