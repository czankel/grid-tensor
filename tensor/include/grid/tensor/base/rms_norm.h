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

#include "matmul.h"

namespace grid {

namespace {
  template <typename> struct Eps {};
  template <> struct Eps<float>  { constexpr static float  default_value = 1e-5f; float  value; };
  template <> struct Eps<double> { constexpr static double default_value = 1e-5f; double value; };
}

/// TensorRmsNorm<Tensor> implements RMS norm.
template <typename T, size_t TRank, PrimitiveTensor TTensor>
class TensorRmsNorm<Tensor, T, TRank, TTensor>
{

 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  constexpr static size_t rank = TRank;

  template <ConvertibleTo<Tensor> T1>
  TensorRmsNorm(T1&& tensor, value_type eps = Eps<value_type>::default_value)
  requires (std::is_floating_point_v<value_type> && rank > 0)
   : tensor_{std::forward<T1>(tensor)},
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
    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
      value += *src * *src;
    return std::tuple{value, count};
  }

  template <size_t N>
  inline auto
  SumSquare(const_pointer src,
            std::span<const size_t,  N> dimensions,
            std::span<const ssize_t, N> strides) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
    value_type value{0};
    size_t count = 0;
    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
    {
      auto [s, c] = SumSquare(src,
                              std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
                              std::span<const ssize_t, N - 1>(strides.begin() + 1, N - 1));
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
    return TensorMatMul(tensor_, Tensor{scale})();
  }

 private:
  TTensor    tensor_;
  value_type eps_;
};

//
// CTAD
//

template <ConvertibleTo<Tensor> TTensor>
TensorRmsNorm(TTensor&&)
  -> TensorRmsNorm<Tensor,
                   typename std::remove_reference_t<TTensor>::value_type,
                   std::remove_reference_t<TTensor>::rank,
                   TTensor&&>;

} // end of namespace grid

#endif // GRID_TENSOR_BASE_RMS_NORN_H
