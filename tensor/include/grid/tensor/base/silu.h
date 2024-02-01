//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_SILU_H
#define GRID_TENSOR_BASE_SILU_H

#include <math.h>
#include <tuple>

#include "matmul.h"

namespace grid {

/// TensorSilu<Tensor> implements RMS norm.
template <typename T, size_t TRank, PrimitiveTensor TTensor>
class TensorSilu<Tensor, T, TRank, TTensor>
{
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  constexpr static size_t rank = TRank;

  template <ConvertibleTo<Tensor> T1>
  TensorSilu(T1&& tensor) : tensor_(std::forward<T1>(tensor)) {}

  // delete assignment and copy/move constructors
  TensorSilu() = delete;
  TensorSilu(const TensorSilu& other) = delete;
  TensorSilu(TensorSilu&& other) = delete;
  TensorSilu& operator=(const TensorSilu& other) = delete;
  TensorSilu& operator=(TensorSilu&& other) = delete;

 private:
  inline void
  Silu(pointer dst,
       const_pointer src,
       std::span<const size_t,  1> dimensions,
       std::span<const ssize_t, 1> strides) const
  {
    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
      dst[i] = *src / (1.0f + exp(-*src));
  }

  template <size_t _N>
  inline void
  Silu(pointer dst,
       const_pointer src,
       std::span<const size_t,  _N> dimensions,
       std::span<const ssize_t, _N> strides) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
      Silu(dst, src,
           std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
           std::span<const ssize_t, _N - 1>(strides.begin() + 1, _N - 1));
  }

 public:
  /// operator()() executes the operation and returns a tensor.
  auto operator()() const requires (std::is_floating_point_v<value_type>)
  {
    // FIXME: temporary necessary?
    auto result = Tensor(tensor_.Dimensions(), Uninitialized<value_type>{});
    Silu(result.Data(),
         tensor_.Data(),
         std::span(tensor_.Dimensions()),
         std::span(tensor_.Strides()));
    return result;
  }

 private:
  TTensor tensor_;
};

//
// CTAD
//

template <ConvertibleTo<Tensor> TTensor>
TensorSilu(TTensor)
  -> TensorSilu<Tensor, typename TTensor::value_type, TTensor::rank, typename to_tensor<TTensor>::type>;

} // end of namespace grid

#endif // GRID_TENSOR_BASE_SILU_H
