//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_MULTIPLY_H
#define GRID_TENSOR_BASE_MULTIPLY_H

namespace grid {

/// TensorMul<Tensor> implements tensor multiplication operation for tensors of the same and
/// different ranks, such as matrix multiplication (MatMul) and vector dot-product (VecDot).
template <typename _Tp, size_t _Rank, PrimitiveTensor _Tensor1, PrimitiveTensor _Tensor2>
class TensorMul<Tensor, _Tp, _Rank, _Tensor1, _Tensor2>
{
 public:
  using value_type = _Tp;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = _Rank;

  template <ConvertibleTo<Tensor> T1, ConvertibleTo<Tensor> T2>
  TensorMul(T1&& tensor1, T2&& tensor2)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {
    if constexpr (_Tensor1::rank > 0 && _Tensor2::rank > 0)
      if (tensor1_.Dimensions()[_Tensor1::rank - 1] != tensor2_.Dimensions()[0])
        throw std::runtime_error("dimensions don't match");
  }

  template <ConvertibleTo<Tensor> T1, Scalar S>
  TensorMul(T1&& tensor1, S scalar)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(scalar)
  {}


  // delete assignment and copy/move constructors
  TensorMul() = delete;
  TensorMul(const TensorMul& other) = delete;
  TensorMul(TensorMul&& other) = delete;
  TensorMul& operator=(const TensorMul& other) = delete;
  TensorMul& operator=(TensorMul&& other) = delete;

 private:
  inline void VecDot(pointer dest, const_pointer src1, const_pointer src2,
                     size_t dimensions, ssize_t strides1, ssize_t strides2) const
  {
    value_type sum{0};
    for (size_t i = 0; i < dimensions; i++)
    {
      sum += *src1 * *src2;
      reinterpret_cast<const char*&>(src1) += strides1;
      reinterpret_cast<const char*&>(src2) += strides2;
    }
    *dest = sum;
  }

  // Note that dimensions are mkn: M_m_k * M_k_n -> M(m,n)
  // Note that strides for all tensors (destination and sources) are:
  //    [0] row: m -> m + 1,  [1] col: n -> n + 1
  inline void MatMul(pointer dest, const_pointer src1, const_pointer src2,
                     std::span<const size_t,  3> dimensions,
                     std::span<const ssize_t, 2> strides0,
                     std::span<const ssize_t, 2> strides1,
                     std::span<const ssize_t, 2> strides2) const
  {
    //printf("strides1 %ld %ld strides2 %ld %ld\n", strides1[0], strides1[1], strides2[0], strides2[1]);
    for (size_t m = 0; m < dimensions[0]; m++)
    {
      pointer destprime = dest;
      const_pointer src2prime = src2;
      for (size_t n = 0; n < dimensions[2]; n++)
      {
        VecDot(destprime, src1, src2prime, dimensions[1], strides1[1], strides2[0]);
        reinterpret_cast<char*&>(destprime) += strides0[1];
        reinterpret_cast<const char*&>(src2prime) += strides2[1];
      }
      reinterpret_cast<char*&>(dest) += strides0[0];
      reinterpret_cast<const char*&>(src1) += strides1[0];
    }
  }

  inline void Scale(pointer dest, const_pointer src, const value_type factor,
                    std::span<const size_t,  1> dimensions,
                    std::span<const ssize_t, 1> strides0,
                    std::span<const ssize_t, 1> strides1) const
  {
    //printf("Scale %lu, %p * x -> %p strides %lu %lu %lu\n", dimensions[0], src, dest, strides0[0], strides1[0], sizeof(*src));
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      *dest = *src * factor;
      reinterpret_cast<char*&>(dest) += strides0[0];
      reinterpret_cast<const char*&>(src) += strides1[0];
    }
  }

  template <size_t _N>
  inline void Scale(pointer dest, const_pointer src, const value_type factor,
                    std::span<const size_t,  _N> dimensions,
                    std::span<const ssize_t, _N> strides0,
                    std::span<const ssize_t, _N> strides1) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      Scale(dest, src, factor,
          std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
          std::span<const ssize_t, _N - 1>(strides0.begin() + 1, _N - 1),
          std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1));
      reinterpret_cast<char*&>(dest) += strides0[0];
      reinterpret_cast<const char*&>(src) += strides1[0];
    }
  }

 public:

  /// operator()() executes the operation and returns a tensor.
  /// VectorDot product
  auto operator()() const requires (_Tensor1::rank == 1 && _Tensor2::rank == 1)
  {
    size_t dimensions = tensor1_.Dimensions()[0];
    auto result = Tensor(Uninitialized<value_type>{});
    //printf("VecDot %p %p -> %p\n", tensor1_.Data(), tensor2_.Data(), result.Data());

    VecDot(result.Data(),
           tensor1_.Data(),
           tensor2_.Data(),
           dimensions,
           tensor1_.Strides()[0],
           tensor2_.Strides()[0]);

    return result;
  }

  /// Matrix x Matrix: (M,I) * (I,N) -> (M,N)
  auto operator()() const requires (_Tensor1::rank == 2 && _Tensor2::rank == 2)
  {
    auto&& dimensions1 = tensor1_.Dimensions();
    auto&& dimensions2 = tensor2_.Dimensions();
    size_t dimensions[] = {dimensions1[0], dimensions1[1], dimensions2[1]};
    auto result = Tensor({dimensions1[0], dimensions2[1]}, Uninitialized<value_type>{});

    // transpose right matrix
    //auto strides = tensor2_.Strides();
    //std::swap(strides[0], strides[1]);

    MatMul(result.Data(),
           tensor1_.Data(),
           tensor2_.Data(),
           std::span(dimensions),
           std::span(result.Strides()),
           std::span(tensor1_.Strides()),
           std::span(tensor2_.Strides()));

    return result;
  }

  /// Matrix/Vector * Scaler
  auto operator()() const requires (_Tensor2::rank == 0)
  {
    auto&& dimensions = tensor1_.Dimensions();
    auto result = Tensor(dimensions, Uninitialized<value_type>{});
    //printf("ScaleMat %p %p -> %p\n", tensor1_.Data(), tensor2_.Data(), result.Data());

    Scale(result.Data(),
          tensor1_.Data(),
          *tensor2_.Data(),
          std::span(dimensions),
          std::span(result.Strides()),
          std::span(tensor1_.Strides()));
    return result;
  }

  /// Scaler * Matrix/Vector
  auto operator()() const requires (_Tensor1::rank == 0 && _Tensor2::rank != 0)
  {
    auto&& dimensions = tensor2_.Dimensions();
    auto result = Tensor(dimensions, Uninitialized<value_type>{});
    //printf("ScaleMat %p %p -> %p\n", tensor1_.Data(), tensor2_.Data(), result.Data());

    Scale(result.Data(),
          tensor2_.Data(),
          *tensor1_.Data(),
          std::span(dimensions),
          std::span(result.Strides()),
          std::span(tensor2_.Strides()));

    return result;
  }

  // Matrix * Vector: M_m_n * V_n -> V_m  -- M_m_n * V_n_1
  auto operator()() const requires (_Tensor1::rank == 2 && _Tensor2::rank == 1)
  {
    auto&& dimensions1 = tensor1_.Dimensions();
    auto result = Tensor(dimensions1[0], Uninitialized<value_type>{});
    size_t dimensions[] = {dimensions1[0], dimensions1[1], 1};

    std::array<ssize_t, 2> strides0 = {result.Strides()[0], 0};
    std::array<ssize_t, 2> strides2 = {tensor2_.Strides()[0], 0};

    MatMul(result.Data(),
           tensor1_.Data(),
           tensor2_.Data(),
           std::span(dimensions),
           std::span(strides0),
           std::span(tensor1_.Strides()),
           std::span(strides2));

    return result;
  }

  // Vector * Matrix: V_m * M_m_n -> V_n  -- V_1_m * M_m_n
  auto operator()() const requires (_Tensor1::rank == 1 && _Tensor2::rank == 2)
  {
    auto&& dimensions2 = tensor2_.Dimensions();
    auto result = Tensor(dimensions2[1], Uninitialized<value_type>{});
    size_t dimensions[] = {1, dimensions2[0], dimensions2[1]};

    //printf("VecMat %p %p -> %p\n", tensor1_.Data(), tensor2_.Data(), result.Data());

    std::array<ssize_t, 2> strides  = {0L, result.Strides()[0]};
    std::array<ssize_t, 2> strides1 = {0L, tensor1_.Strides()[0]};
    MatMul(result.Data(),
           tensor1_.Data(),
           tensor2_.Data(),
           std::span(dimensions),
           std::span(strides),
           std::span(strides1),
           std::span(tensor2_.Strides()));

    return result;
  }

 private:
  _Tensor1 tensor1_;
  _Tensor2 tensor2_;
};

//
// CTAD
//

template <ConvertibleTo<Tensor> _Tensor1, ConvertibleTo<Tensor> _Tensor2>
TensorMul(_Tensor1, _Tensor2)
  -> TensorMul<Tensor, typename _Tensor2::value_type, std::max(_Tensor1::rank, _Tensor2::rank),
               typename to_tensor<_Tensor1>::type, typename to_tensor<_Tensor2>::type>;

template <ConvertibleTo<Tensor> _Tensor, Scalar _Scalar>
TensorMul(_Tensor, _Scalar)
  -> TensorMul<Tensor, typename _Tensor::value_type, _Tensor::rank,
               typename to_tensor<_Tensor>::type, Tensor<typename _Tensor::value_type, 0>>;

// FIXME: add option for scalar, tensor

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_MULTIPLY_H
