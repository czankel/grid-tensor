//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_RMS_NORN_H
#define GRID_TENSOR_BASE_RMS_NORN_H

#include <math.h>
#include <tuple>

#include "multiplication.h"

namespace grid {

/// TensorRmsNorm<Tensor> implements RMS norm.
template <typename _T, size_t _Rank, PrimitiveTensor _Tensor>
class TensorRmsNorm<Tensor, _T, _Rank, _Tensor>
{
 public:
  using value_type = _T;
  using pointer = _T*;
  using const_pointer = const _T*;
  constexpr static size_t rank = _Rank;

  template <ConvertibleTo<Tensor> T1>
  TensorRmsNorm(T1&& tensor) : tensor_(std::forward<T1>(tensor)) {}

  ~TensorRmsNorm() {}

  // delete assignment and copy/move constructors
  TensorRmsNorm() = delete;
  TensorRmsNorm(const TensorRmsNorm& other) = delete;
  TensorRmsNorm(TensorRmsNorm&& other) = delete;
  TensorRmsNorm& operator=(const TensorRmsNorm& other) = delete;
  TensorRmsNorm& operator=(TensorRmsNorm&& other) = delete;

  inline auto
  SumSquare(const_pointer src,
            std::span<const size_t,  1> dims,
            std::span<const ssize_t, 1> strides) const
  {
    value_type value{0};
    size_t count = dims[0];
    for (size_t i = 0; i < dims[0]; i++, reinterpret_cast<const char*&>(src) += strides[0])
      value += *src * *src;
    return std::tuple{value, count};
  }

  template <size_t _N>
  inline auto
  SumSquare(const_pointer src,
            std::span<const size_t,  _N> dims,
            std::span<const ssize_t, _N> strides) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
    value_type value{0};
    size_t count = 0;
    for (size_t i = 0; i < dims[0]; i++, reinterpret_cast<const char*&>(src) += strides[0])
    {
      auto [s, c] = SumSquare(src,
                              std::span<const size_t,  _N - 1>(dims.begin() + 1, _N - 1),
                              std::span<const ssize_t, _N - 1>(strides.begin() + 1, _N - 1));
      value += s;
      count += c;
    }
    return std::tuple{value, count};
  }


  auto operator()() const requires (std::is_floating_point_v<value_type>)
  {
    auto [value, count] = SumSquare(tensor_.Data(),
                                    std::span(tensor_.Dimensions()),
                                    std::span(tensor_.Strides()));

    constexpr value_type eps = std::numeric_limits<value_type>::epsilon();
    value_type scale = 1.0/sqrtf(value / count + eps);
    return TensorMul(tensor_, Tensor{scale})();
  }

 private:
  _Tensor tensor_;
};


// CTAD

template <ConvertibleTo<Tensor> _Tensor>
TensorRmsNorm(_Tensor)
  -> TensorRmsNorm<Tensor, typename _Tensor::value_type, _Tensor::rank, typename to_tensor<_Tensor>::type>;

} // end of namespace grid

#endif // GRID_TENSOR_BASE_RMS_NORN_H
