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
    // optimizations:
    // vec = mat x vec
    // mat = mat x vec(broadcast)
    // mat = vec(broadcast) x mat
    if (strides0[1] <= 1 && strides1[1] <= 1 && strides2[0] <= 1)
    {
      // all tensors are contiguous
      if (strides0[0] - dimensions[1] == 0 &&
          strides1[0] - dimensions[1] == 0 &&
          strides2[1] - dimensions[1] == 0)
      {
        for (size_t i = 0; i < dimensions[0] * dimensions[2]; i++)
        {
          T sum{0};
          for (size_t k = 0; k < dimensions[1]; k++)
            sum += src1[k] * src2[k];
          dest[i] = sum;
        }
      }
      // inner vector dot
      else
      {
        for (size_t m = 0; m < dimensions[0]; m++)
        {
          T* destprime = dest;
          const T* src2prime = src2;
          for (size_t n = 0; n < dimensions[2]; n++)
          {
            T sum{0};
            for (size_t k = 0; k < dimensions[1]; k++)
              sum += src1[k] * src2prime[k];
            *destprime = sum;
            destprime += strides0[1];
            src2prime += strides2[1];
          }
          dest += strides0[0];
          src1 += strides1[0];
        }
      }
    }
    // no optimizations
    else
    {
      for (size_t m = 0; m < dimensions[0]; m++)
      {
        T* destprime = dest;
        const T* src2prime = src2;
        for (size_t n = 0; n < dimensions[2]; n++)
        {
          const T* src1tmp = src1;
          const T* src2tmp = src2prime;
          T sum{0};
          for (size_t i = 0; i < dimensions[1]; i++)
          {
            sum += *src1tmp * *src2tmp;
            src1tmp += strides1[1];
            src2tmp += strides2[0];
          }
          *destprime = sum;
          destprime += strides0[1];
          src2prime += strides2[1];
        }
        dest += strides0[0];
        src1 += strides1[0];
      }
    }
  }

 public:
  // vecdot
  template <typename T>
  void operator()(T* dst, const T* src1, const T* src2,
                  const size_t dimensions, const ssize_t strides1, const ssize_t strides2)
  {
    T sum{0};
    if (strides1 == 1 && strides2 == 1)
    {
      for (size_t i = 0; i < dimensions; i++)
        sum += src1[i] * src2[i];
    }
    else
    {
      for (size_t i = 0; i < dimensions; i++)
      {
        sum += *src1 * *src2;
        src1 += strides1;
        src2 += strides2;
      }
    }
    *dst = sum;
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
