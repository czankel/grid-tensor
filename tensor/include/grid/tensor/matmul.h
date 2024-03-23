//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_MATMUL_H
#define GRID_TENSOR_MATMUL_H

#include <span>
#include <algorithm>
#include <ranges>

#include "concepts.h"

namespace grid {

//
// Device-specific operator
//

template <typename> class MatMulOperator;
template <typename, size_t, typename> class Tensor;

template <AnyTensor TTensor1, AnyTensor TTensor2>
class MatMulFunction
{
  using device = tensor_device_t<TTensor1>;

 public:
  using tensor1_type = std::remove_reference_t<TTensor1>;
  using tensor2_type = std::remove_reference_t<TTensor2>;
  using value_type = std::common_type_t<typename tensor1_type::value_type, typename tensor2_type::value_type>;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  constexpr static size_t tensor1_rank = tensor1_type::rank;
  constexpr static size_t tensor2_rank = tensor2_type::rank;
  // TODO: OK for combination of rank-2 and rank-1 but maybe not rank-3 and higher?
  constexpr static size_t rank =
   tensor1_type::rank != 1 || tensor2_type::rank != 1 ? std::min(tensor1_type::rank, tensor2_type::rank) : 0;

  template <typename T1, typename T2>
  requires (tensor1_rank > 0 && tensor2_rank > 0)
  MatMulFunction(T1&& tensor1, T2&& tensor2)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {
    if constexpr (tensor1_rank > 0 && tensor2_rank > 0)
    {
      // matmul: lhs columns (dim[rank-1])  and rhs rows (dim[rank-2]) have to match; assume column-vector
      size_t dim = tensor2_rank > 1 ? tensor2_rank - 2 : 0;
      size_t lhs = tensor1_.Dimensions()[tensor1_rank - 1];
      size_t rhs = tensor2_.Dimensions()[dim];
      if (lhs != rhs)
        throw std::runtime_error("dimensions don't match: lhs: " + std::to_string(lhs) + " rhs: " + std::to_string(rhs));
    }
  }

  // delete assignment and copy/move constructors
  MatMulFunction() = delete;
  MatMulFunction(const MatMulFunction& other) = delete;
  MatMulFunction& operator=(const MatMulFunction& other) = delete;

  /// operator()() executes and returns a (scalar) tensor with the 'vector dot' multiplication.
  auto operator()() const requires (tensor1_rank == 1 && tensor2_rank == 1)
  {
    if (tensor1_.Dimensions()[0] != tensor2_.Dimensions()[0])
      throw std::runtime_error("mismatching dimensions in vector product");

    auto result = Tensor<value_type, 0, DeviceMemory<device>>{value_type{0}};
    operator_(result, tensor1_, tensor2_);
    return result;
  }

  /// operator()() executes and returns a (matrix) tensor for a mtrix multiplication.
  auto operator()() const requires (tensor1_rank == 2 && tensor2_rank == 2)
  {
    auto&& dims1 = tensor1_.Dimensions();
    auto&& dims2 = tensor2_.Dimensions();
    if (dims1[1] != dims2[0])
      throw std::runtime_error("mismatching dimensions in matrix multiplication");

    auto result = Tensor<value_type, 2, DeviceMemory<device>>({dims1[0], dims2[1]}, Uninitialized<value_type>{});
    operator_(result, tensor1_, tensor2_);
    return result;
  }

  /// operator()() executes and returns a (vector) tensor of a matrix * vector multiplication.
  auto operator()() const requires (tensor1_rank == 2 && tensor2_rank == 1)
  {
    auto&& dims1 = tensor1_.Dimensions();
    auto&& dims2 = tensor2_.Dimensions();
    if (dims1[1] != dims2[0])
      throw std::runtime_error("mismatching dimensions in matrix multiplication");

    auto result = Tensor<value_type, 1, DeviceMemory<device>>(dims1[0], Uninitialized<value_type>{});
    operator_(result, tensor1_, tensor2_);
    return result;
  }

  /// operator()() executes and returns a (vector) tensor of a vector * matrix multiplication.
  auto operator()() const requires (tensor1_rank == 1 && tensor2_rank == 2)
  {
    auto&& dims1 = tensor1_.Dimensions();
    auto&& dims2 = tensor2_.Dimensions();
    if (dims1[0] != dims2[0])
      throw std::runtime_error("mismatching dimensions in matrix multiplication");

    auto result = Tensor<value_type, 1, DeviceMemory<device>>(dims2[1], Uninitialized<value_type>{});
    operator_(result, tensor1_, tensor2_);
    return result;
  }

 private:
  MatMulOperator<device> operator_;
  TTensor1 tensor1_;
  TTensor2 tensor2_;
};

template <typename T1, typename T2> MatMulFunction(T1&&, T2&&)
  -> MatMulFunction<typename to_tensor<T1>::type, typename to_tensor<T2>::type>;

template <TensorConvertible TTensor1, TensorConvertible TTensor2>
requires (std::remove_cvref_t<TTensor1>::rank > 0 && std::remove_cvref_t<TTensor2>::rank > 0)
auto MatMul(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return MatMulFunction(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

} // end of namespace grd

#endif  // GRID_TENSOR_MATMUL_H

