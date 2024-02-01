//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_ELEMMUL_H
#define GRID_TENSOR_BASE_ELEMMUL_H

namespace grid {

/// TensorElemMul<Tensor> implements an element-wise tensor multiplication.
/// This is different from matrix multiplication (MatMul) or the vector dot-product (VecDot).
template <typename T, size_t TRank, PrimitiveTensor TTensor1, PrimitiveTensor TTensor2>
class TensorElemMul<Tensor, T, TRank, TTensor1, TTensor2>
{
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  constexpr static size_t rank = TRank;

  template <ConvertibleTo<Tensor> T1, ConvertibleTo<Tensor> T2>
  TensorElemMul(T1&& tensor1, T2&& tensor2)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {}

  template <ConvertibleTo<Tensor> T1, Arithmetic S>
  TensorElemMul(T1&& tensor1, S scalar)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(scalar)
  {}


  // delete assignment and copy/move constructors
  TensorElemMul() = delete;
  TensorElemMul(const TensorElemMul& other) = delete;
  TensorElemMul(TensorElemMul&& other) = delete;
  TensorElemMul& operator=(const TensorElemMul& other) = delete;
  TensorElemMul& operator=(TensorElemMul&& other) = delete;

 private:

  inline void mul(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  1> dimensions,
                  std::span<const ssize_t, 1> strides0,
                  std::span<const ssize_t, 1> strides1,
                  std::span<const ssize_t, 1> strides2) const
   {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      dest[i] = *src1 * *src2;
      src1 += strides1[0];
      src2 += strides2[0];
    }
  }

  template <size_t N>
  inline void mul(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  N> dimensions,
                  std::span<const ssize_t, N> strides0,
                  std::span<const ssize_t, N> strides1,
                  std::span<const ssize_t, N> strides2) const
  {
    for (size_t m = 0; m < dimensions[0]; m++)
    {
      mul(dest, src1, src2,
          std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides0.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides2.begin() + 1, N - 1));

      dest += strides0[0];
      src1 += strides1[0];
      src2 += strides2[0];
    }
  }

 public:

  /// operator()() executes the element-wise multiplication and returns a new resulting tensor.
  auto operator()() const
  {
    auto dimensions1 = tensor1_.Dimensions();
    auto dimensions2 = tensor2_.Dimensions();

    // handle broadcast
    std::array<size_t, rank> dimensions;
    if constexpr (TTensor1::rank <= TTensor2::rank)
    {
      size_t i = 0;
      for (; i < TTensor2::rank; i++)
        dimensions[i] = dimensions1[i];
      for (size_t j = 0; i < rank; i++, j++)
        dimensions[i] = dimensions1[i] != 1 ? dimensions1[i] : dimensions2[j];
    }
    else
    {
      size_t i = 0;
      for (; i < TTensor1::rank; i++)
        dimensions[i] = dimensions2[i];
      for (size_t j = 0; i < rank; i++, j++)
        dimensions[i] = dimensions2[i] != 1 ? dimensions2[i] : dimensions2[j];
    }

    auto result = Tensor(dimensions, Uninitialized<value_type>{});

    mul(result.Data(),
        tensor1_.Data(),
        tensor2_.Data(),
        std::span<const size_t, rank>(dimensions),
        std::span(result.Strides()),
        std::span(tensor1_.Strides()),
        std::span(tensor2_.Strides()));

    return result;
  }

 private:
  TTensor1 tensor1_;
  TTensor2 tensor2_;
};

//
// CTAD
//

template <ConvertibleTo<Tensor> TTensor1, ConvertibleTo<Tensor> TTensor2>
TensorElemMul(TTensor1, TTensor2)
  -> TensorElemMul<Tensor, typename TTensor2::value_type, std::max(TTensor1::rank, TTensor2::rank),
               typename to_tensor<TTensor1>::type, typename to_tensor<TTensor2>::type>;

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_ELEMMUL_H
