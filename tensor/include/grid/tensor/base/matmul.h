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

/// MatmulOperator implements a multiplication operation for tensors
/// different ranks, such as matrix multiplication (Matmul) and vector dot-product (VecDot).
/// Note that all dimensions are assumed to be correct.
template <> class MatmulOperator<device::Base>
{
  // optimized vector dot multiplication for contiguous vectors.
  template <typename T>
  inline void VecDot(T* dest, const T* src1, const T* src2, const size_t dim) const
  {
    T sum{0};
    for (size_t n = 0; n < dim; n++)
      sum += src1[n] * src2[n];
    dest[0] = sum;
  }

  // default vector dot multiplication for non-contigous vectors.
  template <typename T>
  inline void VecDot(T* dest, const T* src1, const T* src2, const size_t dim,
                     const ssize_t& strides1, const ssize_t& strides2) const
  {
    T sum{0};
    for (size_t n = 0; n < dim; n++)
    {
      sum += src1[0] * src2[0];
      src1 += strides1;
      src2 += strides2;
    }
    dest[0] = sum;
  }

  // optimized mat x vec multiplication for a contiguous matrix and vector.
  template <typename T>
  inline void MatVec(T* dest, const T* src1, const T* src2,
                     const size_t& dim_m, const size_t& dim_n,
                     const ssize_t& strides1) const
  {
    for (size_t m = 0; m < dim_m; m++)
    {
      T sum{0};
      for (size_t n = 0; n < dim_n; n++)
        sum += src1[n] * src2[n];
      dest[m] = sum;
      src1 += strides1;
    }
  }

  // default max x vec multiplication for non-contiguous matrix/vector.
  template <typename T>
  inline void MatVec(T* dest, const T* src1, const T* src2,
                     const size_t& dim_m, const size_t& dim_n,
                     const ssize_t& strides0,
                     const ssize_t& strides1_m,
                     const ssize_t& strides1_n,
                     const ssize_t& strides2) const
  {
    for (size_t m = 0; m < dim_m; m++)
    {
      auto* src1prime = src1;
      auto* src2prime = src2;
      T sum{0};
      for (size_t n = 0; n < dim_n; n++)
      {
        sum += src1prime[0] * src2prime[0];
        src1prime += strides1_n;
        src2prime += strides2;
      }
      dest[0] = sum;
      dest += strides0;
      src1 += strides1_m;
    }
  }

  // optimized vec x mat multiplication for contiguous vector and matrix.
  template <typename T>
  inline void VecMat(T* dest, const T* src1, const T* src2,
                     const size_t& dim_m, const size_t& dim_n,
                     const size_t& strides_n) const
  {
    for (size_t n = 0; n < dim_n; n++)
      dest[n] = 0;

    for (size_t m = 0; m < dim_m; m++, src2 += strides_n)
      for (size_t n = 0; n < dim_n; n++)
        dest[n] += src1[m] * src2[n];
  }

  // matrix multiplication. Note that dimensions are mn,k: M_m_k * M_k_n -> M_m_n

  // contiguous data
  template <typename T>
  inline void Matmul(T* dest, const T* src1, const T* src2,
                     std::span<const size_t,  2> dimensions, size_t dim_k) const
  {
    for (size_t m = 0; m < dimensions[0]; m++)
    {
      const T* src2prime = src2;
      for (size_t n = 0; n < dimensions[1]; n++)
      {
        T sum{0};
        for (size_t k = 0; k < dim_k; k++)
          sum += src1[k] * src2prime[k];
        dest[n] = sum;
        src2prime += dim_k;
      }
      src1 += dim_k;
      dest += dimensions[1];
    }
  }

  // semi-optimized: only lowest 'rank' is contiguous and rhs transposed
  template <typename T>
  inline void Matmul(T* dest, const T* src1, const T* src2,
                     std::span<const size_t,  2> dimensions, size_t dim_k,
                     const ssize_t& strides0, const ssize_t& strides1, const ssize_t& strides2) const
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
        destprime[n] = sum;
        src2prime += strides2;
      }
      dest += strides0;
      src1 += strides1;
    }
  }

  // default unoptimized matrix multiplication
  template <typename T>
  inline void Matmul(T* dest, const T* src1, const T* src2,
                     std::span<const size_t,  2> dimensions,
                     size_t                      dim_k,
                     std::span<const ssize_t, 2> strides0,
                     std::span<const ssize_t, 2> strides1,
                     std::span<const ssize_t, 2> strides2) const
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

    constexpr size_t rank1 = std::ranges::iterator_t<R1>::rank;
    constexpr size_t rank2 = std::ranges::iterator_t<R2>::rank;

    auto& strides0 = result.Strides();
    auto& strides1 = first1.Strides();
    auto& strides2 = first2.Strides();

    // FIXME: use std::moves for strides, dims, etc?

    // mat * mat: M_m_k * M_k_n -> M_m_n
    if constexpr (rank1 == 2 && rank2 == 2)
    {
      size_t dim_k = first1.Extents()[1];
      auto& extents = result.Extents();
      if (strides0[1] <= 1 && strides1[1] <= 1 && strides2[0] <= 1)
      {
        // full optimizations: mat * mat and all tensors are contiguous, strides ignored
        if (strides0[0] - extents[1] == 0 &&
            strides1[0] - dim_k == 0 &&
            strides2[1] - dim_k == 0)
          Matmul(&*result, &*first1, &*first2, std::span(extents), dim_k);

        // semi-contiguous
        else
        {
          Matmul(&*result, &*first1, &*first2, std::span(extents), dim_k,
                 strides0[0], strides1[0], strides2[1]); }
      }
      else
        Matmul(&*result, &*first1, &*first2, std::span(extents), dim_k,
               std::span(strides0), std::span(strides1), std::span(strides2));
    }

    // mat * vec: M_m_n * V_n = M_m_n * V_n_1 -> V_m_1 = V_m
    else if constexpr (rank1 == 2 && rank2 == 1)
    {
      auto& extents = first1.Extents();
      if (strides0[0] <= 1 && strides1[1] <= 1 && strides2[0] == 1)
        MatVec(&*result, &*first1, &*first2, extents[0], extents[1], strides1[0]);
      else
        MatVec(&*result, &*first1, &*first2, extents[0], extents[1],
               strides0[0], strides1[0], strides1[1], strides2[0]);
    }

    // vec * mat: V_m * M_m_n = V_1_m * M_m_n -> V_1_n = V_n (note: pass transposed dims/strides)
    else if constexpr (rank1 == 1 && rank2 == 2)
    {
      auto& extents = first2.Extents();
      if (strides0[0] == 1 && strides1[0] == 1 && strides2[1] == 1)
        VecMat(&*result, &*first1, &*first2, extents[0], extents[1], strides2[0]);
      else if (strides0[0] == 1 && strides1[0] == 1 && strides2[0] == 1)
        MatVec(&*result, &*first2, &*first1, extents[1], extents[0], strides2[1]);
      else
        MatVec(&*result, &*first2, &*first1, extents[1], extents[0],
               strides0[0], strides2[1], strides2[0], strides1[0]);
    }

    // vecdot: V_m * V_m -> scalar
    else if constexpr (rank1 == 1 && rank2 == 1)
    {
      if (strides1[0] == 1 && strides2[0] == 1)
        VecDot(&*result, &*first1, &*first2, first1.Extents()[0]);
      else
        VecDot(&*result, &*first1, &*first2, first1.Extents()[0], strides1[0], strides2[0]);
    }
  }
};

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_MATMUL_H
