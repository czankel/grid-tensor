//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_SOFTMAX_H
#define GRID_TENSOR_BASE_SOFTMAX_H

#include <tuple>
#include <limits>
#include <math.h>

#include "matmul.h"

namespace grid {

/// TensorSoftMax<Tensor> implements softmax of the provided tensor.
template <typename T, size_t TRank, PrimitiveTensor TTensor>
class TensorSoftMax<Tensor, T, TRank, TTensor>
{
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  constexpr static size_t rank = TRank;

  template <ConvertibleTo<Tensor> T1>
  TensorSoftMax(T1&& tensor) : tensor_(std::forward<T1>(tensor)) {}

  // delete assignment and copy/move constructors
  TensorSoftMax() = delete;
  TensorSoftMax(const TensorSoftMax& other) = delete;
  TensorSoftMax(TensorSoftMax&& other) = delete;
  TensorSoftMax& operator=(const TensorSoftMax& other) = delete;
  TensorSoftMax& operator=(TensorSoftMax&& other) = delete;

 private:
  inline auto
  Max(const_pointer src,
      std::span<const size_t,  1> dimensions,
      std::span<const ssize_t, 1> strides) const
  {
    value_type max{std::numeric_limits<value_type>::lowest()};

    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
      max = std::max(max, *src);

    return max;
  }

  template <size_t _N>
  inline auto
  Max(const_pointer src,
      std::span<const size_t,  _N> dimensions,
      std::span<const ssize_t, _N> strides) const
  {
    value_type max{std::numeric_limits<value_type>::lowest()};
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");

    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
    {
      max = std::max(max, Max(src,
                              std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
                              std::span<const ssize_t, _N - 1>(strides.begin() + 1, _N - 1)));
    }
    return max;
  }

  inline auto
  SumExp(pointer dst,
         const_pointer src,
         value_type max,
         std::span<const size_t,  1> dimensions,
         std::span<const ssize_t, 1> strides) const
  {
    value_type sum{0};
    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
    {
      dst[i] = exp(*src - max);
      sum += dst[i];
    }
    return sum;
  }

  template <size_t _N>
  inline auto
  SumExp(pointer dst,
         const_pointer src,
         value_type max,
         std::span<const size_t,  _N> dimensions,
         std::span<const ssize_t, _N> strides) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");

    value_type sum{0};
    for (size_t i = 0; i < dimensions[0]; i++, src += strides[0])
    {
      sum += SumExp(dst, src, max,
                    std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
                    std::span<const ssize_t, _N - 1>(strides.begin() + 1, _N - 1));
    }
    return sum;
  }

 public:
  /// operator()() executes the operation and returns a tensor.
  auto operator()() const requires (std::is_floating_point_v<value_type>)
  {
    // FIXME: really create temporary?? at least comment this is just a 0-rank???
    auto result = Tensor(tensor_.Dimensions(), Uninitialized<value_type>{});
    auto max = Max(tensor_.Data(),
                   std::span(tensor_.Dimensions()),
                   std::span(tensor_.Strides()));

    auto sum = SumExp(result.Data(),
                      tensor_.Data(),
                      max,
                      std::span(tensor_.Dimensions()),
                      std::span(tensor_.Strides()));

    constexpr value_type eps = std::numeric_limits<value_type>::epsilon();  /// FIXME: must be parameter...
    value_type scale = 1.0/(sum + eps);
    return TensorMatMul(result, Tensor{scale})();
  }

 private:
  TTensor tensor_;
};

//
// CTAD
//

template <ConvertibleTo<Tensor> TTensor>
TensorSoftMax(TTensor&&)
  -> TensorSoftMax<Tensor,
                   typename std::remove_cvref_t<TTensor>::value_type,
                   std::remove_cvref_t<TTensor>::rank,
                   typename to_tensor<TTensor>::type>;

} // end of namespace grid

#endif // GRID_TENSOR_BASE_SOFTMAX_H
