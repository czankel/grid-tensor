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
template <typename _Tp, size_t _Rank, PrimitiveTensor _Tensor>
class TensorRmsNorm<Tensor, _Tp, _Rank, _Tensor>
{
  template <typename T> struct Eps {};
  template <> struct Eps<float>  { constexpr static float  default_value = 1e-5f; float  value; };
  template <> struct Eps<double> { constexpr static double default_value = 1e-5f; double value; };

 public:
  using value_type = _Tp;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = _Rank;

  template <ConvertibleTo<Tensor> T1>
  TensorRmsNorm(T1&& tensor, value_type eps = Eps<value_type>::default_value)
  requires (std::is_floating_point_v<value_type> && _Tensor::rank > 0)
   : tensor_(std::forward<T1>(tensor)) ,
     eps_(eps)
  {}

  // delete assignment and copy/move constructors
  TensorRmsNorm() = delete;
  TensorRmsNorm(const TensorRmsNorm& other) = delete;
  TensorRmsNorm(TensorRmsNorm&& other) = delete;
  TensorRmsNorm& operator=(const TensorRmsNorm& other) = delete;
  TensorRmsNorm& operator=(TensorRmsNorm&& other) = delete;

 private:
  inline auto
  SumSquare(const_pointer src,
            std::span<const size_t,  1> dimensions,
            std::span<const ssize_t, 1> strides) const
  {
    value_type value{0};
    size_t count = dimensions[0];
    for (size_t i = 0; i < dimensions[0]; i++, reinterpret_cast<const char*&>(src) += strides[0])
      value += *src * *src;
    return std::tuple{value, count};
  }

  template <size_t _N>
  inline auto
  SumSquare(const_pointer src,
            std::span<const size_t,  _N> dimensions,
            std::span<const ssize_t, _N> strides) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
    value_type value{0};
    size_t count = 0;
    for (size_t i = 0; i < dimensions[0]; i++, reinterpret_cast<const char*&>(src) += strides[0])
    {
      auto [s, c] = SumSquare(src,
                              std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
                              std::span<const ssize_t, _N - 1>(strides.begin() + 1, _N - 1));
      value += s;
      count += c;
    }
    return std::tuple{value, count};
  }

 public:

  /// operator()() executes and returns a tensor with the RMS norm of the stored vector.
  auto operator()() const
  {
    auto [value, count] = SumSquare(tensor_.Data(),
                                    std::span(tensor_.Dimensions()),
                                    std::span(tensor_.Strides()));

    value_type scale = 1.0f/sqrtf(value / count + eps_);
    return TensorMul(tensor_, Tensor{scale})();
  }

 private:
  _Tensor    tensor_;
  value_type eps_;
};

//
// CTAD
//

template <ConvertibleTo<Tensor> _Tensor>
TensorRmsNorm(_Tensor)
  -> TensorRmsNorm<Tensor, typename _Tensor::value_type, _Tensor::rank, typename to_tensor<_Tensor>::type>;

} // end of namespace grid

#endif // GRID_TENSOR_BASE_RMS_NORN_H
