//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_SOFTMAX_FUNCTION_H
#define GRID_TENSOR_SOFTMAX_FUNCTION_H

#include <algorithm>
#include <ranges>
#include <span>

#include "concepts.h"

namespace grid {

//
// Device-specific operator
//

template <typename> class SoftmaxOperator;

/// @brief Softmax implements a lazily executed softmax function.
///
///  @tparm TTensor  tensor
///
template <AnyTensor TTensor>
class Softmax
{
  using device = tensor_device_t<TTensor>;
  static constexpr SoftmaxOperator<device> Operator;

 public:
  using tensor_type = std::remove_reference_t<TTensor>;
  using value_type = std::common_type_t<typename tensor_type::value_type>;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  constexpr static size_t rank = tensor_type::rank;

  template <typename T>
  Softmax(T&& tensor) : tensor_(std::forward<T>(tensor)) {}
  ~Softmax() {}

  // delete assignment and copy/move constructors
  Softmax() = delete;
  Softmax(const Softmax& other) = delete;
  Softmax& operator=(const Softmax& other) = delete;

  /// operator()() evaluates the unary operator and returns a tensor.
  auto operator()() const
  {
    auto& dimensions = tensor_.Dimensions();
    auto result = Tensor(dimensions, Uninitialized<value_type>{});
    Operator(result, tensor_, dimensions, result.Strides(), tensor_.Strides());
    return result;
  }

 private:
  TTensor tensor_;
};

template <typename T> Softmax(T&&) -> Softmax<typename to_tensor<T>::type>;

} // end of namespace grd

#endif  // GRID_TENSOR_SOFTMAX_FUNCTION_H
