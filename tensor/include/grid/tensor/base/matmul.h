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

#include "../device.h"

namespace grid {

/// MatMulOperator implements a multiplication operation for tensors
/// different ranks, such as matrix multiplication (MatMul) and vector dot-product (VecDot).
/// Note that all dimensions are assumed to be correct.
template <> class MatMulOperator<device::Base>
{
  // rank-2 o rank-2
  // Note that dimensions are mn,k: M_m_k * M_k_n -> M_m_n
  template <typename T>
  inline void MatMul(T* dest, const T* src1, const T* src2,
                     std::span<const size_t,  2> dimensions,
                     size_t                      dim_k,
                     std::span<const ssize_t, 2> strides0,
                     std::span<const ssize_t, 2> strides1,
                     std::span<const ssize_t, 2> strides2) const
  {
    // optimizations:
    // mat = mat x mat_T           (strides: 1,1,1)
    // mat = mat x vec (broadcast) (strides: 1,1,0)
    // mat = vec (broadcast) x mat (strides: 1,0,1)
    if (strides0[1] <= 1 && strides1[1] <= 1 && strides2[0] <= 1)
    {
      // full optimizations: mat * mat and all tensors are contiguous, strides ignored
      if (strides0[0] - dimensions[1] == 0 &&
          strides1[0] - dimensions[1] == 0 &&
          strides2[1] - dimensions[1] == 0)
      {
        for (size_t i = 0; i < dimensions[0] * dimensions[1]; i++)
        {
          T sum{0};
          for (size_t k = 0; k < dim_k; k++)
            sum += src1[k] * src2[k];
          dest[i] = sum;
        }
      }

      // optimize inner vector dot: add strides for each (m,n)
      // TODO: could optimize further
      else
      {
        for (size_t m = 0; m < dimensions[0]; m++)
        {
          T* destprime = dest;
          const T* src2prime = src2;
          for (size_t n = 0; n < dimensions[1]; n++)
          {
            T sum{0};
            for (size_t k = 0; k < dim_k; k++)
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

    // no optimizations: add strides for each (m,k,n)
    else
    {
      for (size_t m = 0; m < dimensions[0]; m++)
      {
        T* destprime = dest;
        const T* src2prime = src2;
        for (size_t n = 0; n < dimensions[1]; n++)
        {
          const T* src1tmp = src1;
          const T* src2tmp = src2prime;
          T sum{0};
          for (size_t i = 0; i < dim_k; i++)
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

  // mat x mat
  template<std::ranges::input_range R1,
           std::ranges::input_range R2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<O>> &&
           std::indirectly_copyable<std::ranges::iterator_t<R2>, std::ranges::iterator_t<O>>
  void operator()(O&& o, R1&& r1, R2&& r2) const
  {
    auto first1 = std::ranges::cbegin(r1);
    auto first2 = std::ranges::cbegin(r2);
    auto result = std::ranges::begin(o);

    // mat * mat: M_m_k * M_k_n -> M_m_n
    if constexpr (std::ranges::iterator_t<R1>::rank == 2 && std::ranges::iterator_t<R2>::rank == 2)
      MatMul(&*result, &*first1, &*first2,
             std::span<const size_t,  2>(result.Extents()),
             first1.Extents()[1],
             std::span<const ssize_t, 2>(result.Strides()),
             std::span<const ssize_t, 2>(first1.Strides()),
             std::span<const ssize_t, 2>(first2.Strides()));

    // mat * vec: M_m_n * V_n = M_m_n * V_n_1 -> V_m_1 = V_m
    else if constexpr (std::ranges::iterator_t<R1>::rank == 2 && std::ranges::iterator_t<R2>::rank == 1)
    {
      MatMul(&*result, &*first1, &*first2,
             std::array<size_t, 2>{first1.Extents()[0], 1},
             first1.Extents()[1],
             std::array<const ssize_t, 2>{result.Strides()[0], 0},
             std::span<const ssize_t, 2>(first1.Strides()),
             std::array<const ssize_t, 2>{first2.Strides()[0], 0});
    }

    // vec * mat: V_m * M_m_n = V_1_m * M_m_n -> V_1_n = V_n
    else if constexpr (std::ranges::iterator_t<R1>::rank == 1 && std::ranges::iterator_t<R2>::rank == 2)
      MatMul(&*result, &*first1, &*first2,
             std::array<size_t, 2>{1, first2.Extents()[1]},
             first2.Extents()[0],
             std::array<const ssize_t, 2>{0, result.Strides()[0]},
             std::array<const ssize_t, 2>{0, first1.Strides()[0]},
             std::span<const ssize_t, 2>(first2.Strides()));

    // vec * vec: V_m * V_m -> scalar
    else if constexpr (std::ranges::iterator_t<R1>::rank == 1 && std::ranges::iterator_t<R2>::rank == 1)
    {
      MatMul(&*result, &*first1, &*first2,
             std::array<size_t, 2>{1, 1},
             first1.Extents()[0],
             std::array<const ssize_t, 2>{0, 0},
             std::array<const ssize_t, 2>{0, first1.Strides()[0]},
             std::array<const ssize_t, 2>{0, first2.Strides()[0]});
    }
  }
};


} // end of namespace grid

#endif  // GRID_TENSOR_BASE_MATMUL_H
