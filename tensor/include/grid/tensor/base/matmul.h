//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_MATMUL_H
#define GRID_TENSOR_BASE_MATMUL_H

namespace grid {


/// MatMulOperator implements a multiplication operation for tensors
/// different ranks, such as matrix multiplication (MatMul) and vector dot-product (VecDot).
class MatMulOperator
{
 private:
  template <typename T>
  inline void VecDot(T* dest, const T* src1, const T* src2,
                     size_t dimensions, ssize_t strides1, ssize_t strides2) const
  {
    T sum{0};
    for (size_t i = 0; i < dimensions; i++)
    {
      sum += *src1 * *src2;
      src1 += strides1;
      src2 += strides2;
    }
    *dest = sum;
  }

  // Note that dimensions are mkn: M_m_k * M_k_n -> M(m,n)
  // Note that strides for all tensors (destination and sources) are:
  //    [0] row: m -> m + 1,  [1] col: n -> n + 1
  template <typename T>
  inline void MatMul(T* dest, const T* src1, const T* src2,
                     std::span<const size_t,  3> dimensions,
                     std::span<const ssize_t, 2> strides0,
                     std::span<const ssize_t, 2> strides1,
                     std::span<const ssize_t, 2> strides2) const
  {
    for (size_t m = 0; m < dimensions[0]; m++)
    {
      T* destprime = dest;
      const T* src2prime = src2;
      for (size_t n = 0; n < dimensions[2]; n++)
      {
        VecDot(destprime, src1, src2prime, dimensions[1], strides1[1], strides2[0]);
        destprime += strides0[1];
        src2prime += strides2[1];
      }
      dest += strides0[0];
      src1 += strides1[0];
    }
  }

 public:
  // vecdot
  template <typename T>
  void operator()(T* dst, const T* src1, const T* src2,
                  const size_t dimensions, const ssize_t strides1, const ssize_t strides2)
  {
    VecDot(dst, src1, src2, dimensions, strides1, strides2);
  }

  // matmul
  template <typename T, size_t TRank>
  requires (TRank > 1)
  void operator()(T* dst, const T* src1, const T* src2,
                  const std::array<size_t, TRank+1>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1,
                  const std::array<ssize_t, TRank>& strides2)
  {
    MatMul(dst, src1, src2,
      std::span<const size_t,  TRank+1>(dimensions),
      std::span<const ssize_t, TRank>(strides0),
      std::span<const ssize_t, TRank>(strides1),
      std::span<const ssize_t, TRank>(strides2));
  }
};


} // end of namespace grid

#endif  // GRID_TENSOR_BASE_MATMUL_H
