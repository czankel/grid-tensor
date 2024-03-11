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
#include "../device.h"

namespace grid {
// FIXME: template arg T not used
/// UnaryOperator<Operator> implements element-wise unary operation on a tensors.
///
///  @tparm TOperator binary operator
template <template <typename> typename TOperator, typename T>
class UnaryOperator<TOperator<device::Base>, T>
{
 private:
  // operation on a single element
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src,
                  std::span<const size_t,  0> dimensions,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    TOperator<device::Base>()(dest, src);
  }

  // operation on a single dimension (unoptimized)
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src,
                   std::span<const size_t,  1> dimensions,
                   std::span<const ssize_t, 1>,
                   std::span<const ssize_t, 1> strides1) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      TOperator<device::Base>()(dest + i, src);
      src += strides1[0];
    }
  }

  // operation on higher dimensions (unoptimized)
  template <size_t N, typename const_pointer, typename pointer> inline
  //template <size_t N> inline
  void eval(pointer dest, const_pointer src,
            std::span<const size_t,  N> dimensions,
            std::span<const ssize_t, N> strides0,
            std::span<const ssize_t, N> strides1) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      eval(dest, src,
           std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
           std::span<const ssize_t, N - 1>(strides0.begin() + 1, N - 1),
           std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1));

      dest += strides0[0];
      src += strides1[0];
    }
  }

 public:
  template <typename TTensor0, typename TTensor1, size_t TRank>
  void operator()(TTensor0& result, const TTensor1& tensor,
                  const std::array<size_t, TRank>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1) const
  {
    typename TTensor0::pointer dst = result.Data();
    typename TTensor1::const_pointer src = tensor.Data();
    eval(dst, src,
         std::span<const size_t, TRank>(dimensions),
         std::span<const ssize_t, TRank>(strides0),
         std::span<const ssize_t, TRank>(strides1));
  }
};


//
// Operators
//

template <> class CopyOperator<device::Base>
{
  // scalar X scalar
  template<typename T>
  static inline void operator()(T* dest, const T* src) { *dest = *src; }
};

template <> class NegOperator<device::Base>
{
  // scalar X scalar
  template<typename T>
  static inline void eval(T* dest, const T* src) { *dest = -*src; }
};


} // end of namespace grid

#endif // GRID_TENSOR_BASE_UNARY_H
