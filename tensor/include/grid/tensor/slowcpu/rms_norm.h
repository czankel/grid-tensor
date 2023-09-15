//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_SLOWCPU_RMS_NORN_H
#define GRID_TENSOR_SLOWCPU_RMS_NORN_H

#include <math.h>
#include <tuple>

#include "multiplication.h"

namespace grid {

/// TensorRmsNorm<TensorSlowCpu> implements RMS norm.
template <typename _T, size_t _Rank, TensorFor<TensorSlowCpu> _Tensor>
struct TensorRmsNorm<TensorSlowCpu, _T, _Rank, _Tensor> : TensorBaseOp
{
  using tensor_type = TensorSlowCpu<_T, _Rank>;
  using value_type = _T;

  template <ConvertibleTensorFor<TensorSlowCpu> T1>
  TensorRmsNorm(T1&& tensor) : tensor_(std::forward<T1>(tensor)) {}

  ~TensorRmsNorm() {}

  // delete assignment and copy/move constructors
  TensorRmsNorm() = delete;
  TensorRmsNorm(const TensorRmsNorm& other) = delete;
  TensorRmsNorm(TensorRmsNorm&& other) = delete;
  TensorRmsNorm& operator=(const TensorRmsNorm& other) = delete;
  TensorRmsNorm& operator=(TensorRmsNorm&& other) = delete;

  constexpr static size_t Rank()                          { return _Rank; }

  inline auto
  SumSquare(const char* src,
            std::span<const size_t,  1> dims,
            std::span<const ssize_t, 1> strides) const
  {
    value_type value{0};
    size_t count = dims[0];
    for (size_t i = 0; i < dims[0]; i++, src += strides[0])
    {
      value_type val = *reinterpret_cast<const value_type*>(src);
      value += val * val;
    }
    return std::tuple{value, count};
  }

  template <size_t _N>
  inline auto
  SumSquare(const char* src,
            std::span<const size_t,  _N> dims,
            std::span<const ssize_t, _N> strides) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
    value_type value{0};
    size_t count = 0;
    for (size_t i = 0; i < dims[0]; i++, src += strides[0])
    {
      auto [s, c] = SumSquare(src,
                              std::span<const size_t,  _N - 1>(dims.begin() + 1, _N - 1),
                              std::span<const ssize_t, _N - 1>(strides.begin() + 1, _N - 1));
      value += s;
      count += c;
    }
    return std::tuple{value, count};
  }


  // Functor
  // FIXME limit to floating point? inline std::enable_if_t<!std::is_floating_point_v<_T>...
  tensor_type operator()() const
  {
    auto [value, count] = SumSquare(reinterpret_cast<const char*>(tensor_.Data()),
                                    std::span(tensor_.Dims()),
                                    std::span(tensor_.Strides()));

    constexpr _T eps = std::numeric_limits<_T>::epsilon();
    double scale = 1.0/sqrtf(value / count + eps);
    return TensorMul(tensor_, TensorSlowCpu<double, 0>{scale});
  }

  _Tensor tensor_;
};


// CTAD

template <ConvertibleTensorFor<TensorSlowCpu> _Tensor>
TensorRmsNorm(_Tensor)
  -> TensorRmsNorm<TensorSlowCpu, typename _Tensor::value_type, _Tensor::Rank(), typename _Tensor::tensor_type>;

} // end of namespace grid

#endif // GRID_TENSOR_SLOWCPU_RMS_NORN_H