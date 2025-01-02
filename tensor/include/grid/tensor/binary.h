//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_BINARY_H
#define GRID_TENSOR_BINARY_H

#include <span>
#include <algorithm>
#include <ranges>

#include "concepts.h"
#include "tensor_operator.h"

namespace grid {

/// BinaryOperator is a empty definition for device-specific operators.
template <template <typename> typename, typename> class BinaryOperator;
template <typename, size_t, typename> class Tensor;


//
// Binary Operators
//

template <typename> struct AddOperator;
template <typename> struct SubOperator;
template <typename> struct MulOperator;
template <typename> struct DivOperator;


/// @brief Binary is a wrapper for a device-specific binary operator implementation
///
/// Binary provides a lazy-implementation that only stores the tensors and evaluates
/// the operation with operator().
///
/// The dimensions of the tensors must match following broadcasting rules.
/// The resulting rank is the maximum of the tensor ranks.
///
///  @tparm TOperator binary operator
///  @tparm TTensor1  first tensor
///  @tparm TTensor2  second tensor
///
/// Broadcasting rules:
///
///  - Lower ranking tensors are extended and filled with dimension 1 and stride 0 on the left.
///  - Staring from the right, the dimensions must be either identical or 1.
///
/// Examples:
///   shape: 3, 4, 4 <op> shape: 1, 4, 1    -> OK
///   shape:    4, 1 <op> shape: 3, 4, 3    -> OK
///   shape: 3, 4, 4 <op> shape: 3, 5, 1    -> Error
///
template <typename TOperator, AnyTensor TTensor1, AnyTensor TTensor2>
class Binary : public TensorOperator<std::common_type_t<typename std::remove_cvref_t<TTensor1>::value_type,
                                                        typename std::remove_cvref_t<TTensor2>::value_type>,
                                     std::max(std::remove_cvref_t<TTensor1>::rank, std::remove_cvref_t<TTensor2>::rank),
                                     Binary<TOperator, TTensor1, TTensor2>>

{
 public:
  using typename Binary::TensorOperator::value_type;
  using Binary::TensorOperator::rank;

  template <typename T1, typename T2>
  Binary(TOperator, T1&& tensor1, T2&& tensor2)
   : TensorOperator<value_type, rank, Binary<TOperator, TTensor1, TTensor2>>(*this),
     tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {}

  ~Binary() {}

  // delete assignment and copy/move constructors
  Binary() = delete;
  Binary(const Binary& other) = delete;
  Binary& operator=(const Binary& other) = delete;

 public:

  /// operator()() evaluates the binary operator and returns a tensor.
  auto operator()() const
  {
    auto dimensions = BroadcastDimensions(tensor1_, tensor2_);
    using ResultTensor = Tensor<value_type, rank, DeviceMemory<tensor_device_t<TTensor1>>>;
    auto result = ResultTensor(dimensions, Uninitialized<value_type>{});

    operator_(tensor1_, tensor2_, result);

    return result;
  }

  /// Rank returns the rank of the tensor.
  size_t Rank() const                                     { return rank; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, rank>& Dimensions() const      { return tensor1_.dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, rank>& Strides() const        { return tensor1_.strides_; }

  /// Size returns the data buffer size.
  size_t Size() const                                     { return tensor1_.size_; }

 private:
  static TOperator operator_;
  TTensor1 tensor1_;
  TTensor2 tensor2_;
};

template <typename TOp, typename T1, typename T2> Binary(TOp, T1&&, T2&&)
  -> Binary<TOp, typename to_tensor<T1>::type, typename to_tensor<T2>::type>;


template <typename TOperator, AnyTensor TTensor1, AnyTensor TTensor2>
TOperator Binary<TOperator, TTensor1, TTensor2>::operator_;

//
// Exported binary functions
//

/// @brief Add adds two tensors element-wise (lazily).
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto Add(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return Binary(BinaryOperator<AddOperator, tensor_device_t<TTensor1>>(),
      std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

/// @brief Sub subtracts two tensors element-wise (lazily).
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto Sub(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return Binary(BinaryOperator<SubOperator, tensor_device_t<TTensor1>>(),
      std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

/// @brief Mul multiplies two tensors element-wise (lazily).
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto Mul(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return Binary(BinaryOperator<MulOperator, tensor_device_t<TTensor1>>(),
      std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

/// @brief Mul multiplies a tensors with a scalar.
template <TensorConvertible TTensor, Arithmetic T>
auto Mul(TTensor&& tensor, T scalar)
{
  return Binary(BinaryOperator<MulOperator, tensor_device_t<TTensor>>(),
      std::forward<TTensor>(tensor), Tensor(scalar));
}

/// @brief Mul multiplies a scalar with a tensors.
template <Arithmetic T, TensorConvertible TTensor>
auto Mul(T scalar, TTensor&& tensor)
{
  return Binary(BinaryOperator<MulOperator, tensor_device_t<TTensor>>(),
      std::forward<TTensor>(tensor), Tensor(scalar));
}

/// @brief Div divides a first tensor with a second tensors element-wise (lazily).
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto Div(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return Binary(BinaryOperator<DivOperator, tensor_device_t<TTensor1>>(),
      std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

/// @brief Div divides a tensor by a scalar
template <TensorConvertible TTensor, Arithmetic T>
auto Div(TTensor&& tensor, T scalar)
{
  return Binary(BinaryOperator<DivOperator, tensor_device_t<TTensor>>(),
      std::forward<TTensor>(tensor), Tensor(scalar));
}


} // end of namespace grd

#endif  // GRID_TENSOR_BINARY_H
