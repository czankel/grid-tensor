//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_UNARY_FUNCTION_H
#define GRID_TENSOR_UNARY_FUNCTION_H

#include <algorithm>
#include <ranges>
#include <span>

#include "concepts.h"
#include "transform.h"

namespace grid {

//
// Operators
//

template <typename> class CopyOperator;
template <typename> class NegOperator;

/// All UnaryOperator ... FIXME
template <typename, typename> class UnaryOperator;

/// @brief Unary<Operator> implements lazy element-wise unary operations on a tensors.
///
/// Unary stores the tensor (or tensor operator) and evaluates it on execution with
/// the operator().
///
///  @tparm TOperator unary operator
///  @tparm TTensor  second tensor
///
template <template <typename> typename TOperator, AnyTensor TTensor>
class Unary
{
  using device = tensor_device_t<TTensor>;

 public:
  using tensor_type = std::remove_reference_t<TTensor>;
  using value_type = std::common_type_t<typename tensor_type::value_type>;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  constexpr static size_t rank = tensor_type::rank;

  template <typename T>
  Unary(TOperator<device>, T&& tensor) : tensor_(std::forward<T>(tensor)) {}
  ~Unary() {}

  // delete assignment and copy/move constructors
  Unary() = delete;
  Unary(const Unary& other) = delete;
  Unary& operator=(const Unary& other) = delete;

 public:

  /// operator()() evaluates the unary operator and returns a tensor.
  auto operator()() const
  {
    auto& dimensions = tensor_.Dimensions();
    auto result = Tensor(dimensions, Uninitialized<value_type>{});
    operator_(result, tensor_, dimensions, result.Strides(), tensor_.Strides());
    return result;
  }

 private:
  UnaryOperator<TOperator<device>, value_type> operator_;
  TTensor tensor_;
};

template <template <typename> typename TOp, typename T, typename TDev>
Unary(TOp<TDev>, T&&) -> Unary<TOp, typename to_tensor<T>::type>;


//
// Exported unary functions
//

/// @brief Copy returns a copy of the tensor.
template <TensorConvertible TTensor> auto Copy(TTensor&& tensor)
{
  return Unary(CopyOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}

/// @brief Neg returns a tensor with each element negatd.
template <TensorConvertible TTensor> auto Neg(TTensor&& tensor)
{
  return Unary(NegOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}
#if 0
// FIXME: how to integrate this?
/// @brief Softmax ... FIXME
template <TensorConvertible TTensor> auto Softmax(TTensor&& tensor)
{
  return Unary(SoftmaxOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}
#endif

} // end of namespace grd

#endif  // GRID_TENSOR_UNARY_FUNCTION_H
