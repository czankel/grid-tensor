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

// TODO: remove when operators are device templated
#include "base/matmul.h"

namespace grid {

template <typename, size_t, typename> class Tensor;

template <AnyTensor TTensor1, AnyTensor TTensor2>
class MatMulFunction
{
 public:
  using tensor1_type = std::remove_reference_t<TTensor1>;
  using tensor2_type = std::remove_reference_t<TTensor2>;
  using value_type = std::common_type_t<typename tensor1_type::value_type, typename tensor2_type::value_type>;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  constexpr static size_t tensor1_rank = tensor1_type::rank;
  constexpr static size_t tensor2_rank = tensor2_type::rank;
  // TODO: OK for combination of rank-2 and rank-1 but maybe not rank-3 and higher?
  constexpr static size_t rank = std::min(tensor1_type::rank, tensor2_type::rank);

  template <typename T1, typename T2>
  requires (tensor1_rank > 0 && tensor2_rank > 0)
  MatMulFunction(T1&& tensor1, T2&& tensor2)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {
    if constexpr (tensor1_rank > 0 && tensor2_rank > 0)
    {
      // matmul: dim-m (rank-2) and dim-n (rank-1) have to match; vectors have only one dim
      size_t dim_n = tensor1_rank > 1 ? tensor1_rank - 2 : 0;
      if (tensor1_.Dimensions()[tensor1_rank-1] != tensor2_.Dimensions()[dim_n])
        throw std::runtime_error("dimensions don't match");
    }
  }

  // delete assignment and copy/move constructors
  MatMulFunction() = delete;
  MatMulFunction(const MatMulFunction& other) = delete;
  MatMulFunction(MatMulFunction&& other) = delete;
  MatMulFunction& operator=(const MatMulFunction& other) = delete;
  MatMulFunction& operator=(MatMulFunction&& other) = delete;


  /// operator()() executes and returns a (scalar) tensor with the 'vector dot' multiplication.
  auto operator()() const requires (tensor1_rank == 1 && tensor2_rank == 1)
  {
    auto result = value_type{0};

    std::invoke(MatMulOperator{},
                &result,
                tensor1_.Data(),
                tensor2_.Data(),
                tensor1_.Dimensions()[0],
                tensor1_.Strides()[0],
                tensor2_.Strides()[0]);

    return Tensor{result};
  }

  /// operator()() executes and returns a (matrix) tensor for a mtrix multiplication.
  auto operator()() const requires (tensor1_rank == 2 && tensor2_rank == 2)
  {
    auto&& dimensions1 = tensor1_.Dimensions();
    auto&& dimensions2 = tensor2_.Dimensions();
    std::array<size_t, 3> dimensions{dimensions1[0], dimensions1[1], dimensions2[1]};
    auto result = Tensor({dimensions1[0], dimensions2[1]}, Uninitialized<value_type>{});

    std::invoke(MatMulOperator{},
                result.Data(),
                tensor1_.Data(),
                tensor2_.Data(),
                dimensions,
                result.Strides(),
                tensor1_.Strides(),
                tensor2_.Strides());

    return result;
  }

  /// operator()() executes and returns a (vector) tensor of a matrix * vector multiplication.
  auto operator()() const requires (tensor1_rank == 2 && tensor2_rank == 1)
  {
    auto&& dimensions1 = tensor1_.Dimensions();
    auto result = Tensor(dimensions1[0], Uninitialized<value_type>{});

    std::array<size_t,  3> dimensions{1, dimensions1[0], dimensions1[1]};
    std::array<ssize_t, 2> strides0{0L, result.Strides()[0]};
    std::array<ssize_t, 2> strides2{0L, tensor2_.Strides()[0]};

    // Use: M_m_n * V_n = M_m_n * V_n_1 -> V_m
    std::invoke(MatMulOperator{},
                result.Data(), tensor1_.Data(), tensor2_.Data(),
                dimensions, strides0, tensor1_.Strides(), strides2);

    return result;
  }

  /// operator()() executes and returns a (vector) tensor of a vector * matrix multiplication.
  auto operator()() const requires (tensor1_rank == 1 && tensor2_rank == 2)
  {
    auto&& dimensions2 = tensor2_.Dimensions();
    auto result = Tensor(dimensions2[1], Uninitialized<value_type>{});

    std::array<size_t,  3> dimensions{1, dimensions2[0], dimensions2[1]};
    std::array<ssize_t, 2> strides0{0L, result.Strides()[0]};
    std::array<ssize_t, 2> strides1{0L, tensor1_.Strides()[0]};

    // Use V_m * M_m_n = V_1_m * M_m_n -> V_n
    std::invoke(MatMulOperator{},
                result.Data(), tensor1_.Data(), tensor2_.Data(),
                dimensions, strides0, strides1, tensor2_.Strides());

    return result;
  }


 private:
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

