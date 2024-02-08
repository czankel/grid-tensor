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

// TODO: remove when operators are device templated
#include "base/unary_operators.h"
#include "base/rms_norm.h"

namespace grid {

template <typename, size_t, typename> class Tensor;
template <PrimitiveTensor, size_t> class TensorView;


/// @brief UnaryFunction<Operator> implements lazy element-wise unary operations on a tensors.
///
/// UnaryFunction stores the tensor (or tensor operator) and evaluates it on execution with
/// the operator().
///
///  @tparm TOperator unary operator
///  @tparm TTensor  second tensor
///
template <typename TOperator, AnyTensor TTensor>
class UnaryFunction
{
 public:
  using tensor_type = std::remove_reference_t<TTensor>;
  using value_type = std::common_type_t<typename tensor_type::value_type>;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  constexpr static size_t rank = tensor_type::rank;

  template <typename T>
  UnaryFunction(TOperator, T&& tensor) : tensor_(std::forward<T>(tensor)) {}
  ~UnaryFunction() {}

  // delete assignment and copy/move constructors
  UnaryFunction() = delete;
  UnaryFunction(const UnaryFunction& other) = delete;
  UnaryFunction(UnaryFunction&& other) = delete;
  UnaryFunction& operator=(const UnaryFunction& other) = delete;
  UnaryFunction& operator=(UnaryFunction&& other) = delete;

 public:

  /// operator()() evaluates the unary operator and returns a tensor.
  auto operator()() const
  {
    auto& dimensions = tensor_.Dimensions();
    auto result = Tensor(dimensions, Uninitialized<value_type>{});
    TOperator{}(result.Data(), tensor_.Data(), dimensions, result.Strides(), tensor_.Strides());
    return result;
  }

 private:
  TTensor tensor_;
};

template <typename TOp, typename T> UnaryFunction(TOp, T&&) -> UnaryFunction<TOp, typename to_tensor<T>::type>;

//
// Exported unary functions
//

/// @brief Copy returns a copy of the tensor.
template <TensorConvertible TTensor>
auto Copy(TTensor&& tensor)
{
  return UnaryFunction(UnaryOperator<CopyOperator>{}, std::forward<TTensor>(tensor));
}


/// @brief RmsNorm returns a tensor of the RMS normalized tensor.
template <TensorConvertible TTensor>
auto RmsNorm(TTensor&& tensor)
{
  return UnaryFunction(RmsNormOperator{}, std::forward<TTensor>(tensor));
}


} // end of namespace grd

#endif  // GRID_TENSOR_UNARY_FUNCTION_H
