//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_MATMULH
#define GRID_TENSOR_BASE_MATMULH

namespace grid {

/// TensorMatMul<Tensor> implements tensor multiplication operation for tensors of the same and
/// different ranks, such as matrix multiplication (MatMul) and vector dot-product (VecDot).
template <typename T, size_t TRank, PrimitiveTensor TTensor1, PrimitiveTensor TTensor2>
class TensorMatMul<Tensor, T, TRank, TTensor1, TTensor2>
{
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  constexpr static size_t rank = TRank;
  constexpr static size_t tensor1_rank = std::remove_cvref_t<TTensor1>::rank;
  constexpr static size_t tensor2_rank = std::remove_cvref_t<TTensor2>::rank;

  template <ConvertibleTo<Tensor> T1, ConvertibleTo<Tensor> T2>
  TensorMatMul(T1&& tensor1, T2&& tensor2)
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

  template <ConvertibleTo<Tensor> T1, Arithmetic S>
  TensorMatMul(T1&& tensor1, S scalar)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(scalar)
  {}


  // delete assignment and copy/move constructors
  TensorMatMul() = delete;
  TensorMatMul(const TensorMatMul& other) = delete;
  TensorMatMul(TensorMatMul&& other) = delete;
  TensorMatMul& operator=(const TensorMatMul& other) = delete;
  TensorMatMul& operator=(TensorMatMul&& other) = delete;

 private:
  inline void VecDot(pointer dest, const_pointer src1, const_pointer src2,
                     size_t dimensions, ssize_t strides1, ssize_t strides2) const
  {
    value_type sum{0};
    for (size_t i = 0; i < dimensions; i++)
    {
      sum += *src1 * *src2;
      src1 += strides1;
      src2 += strides2;
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
    for (size_t m = 0; m < dimensions[0]; m++)
    {
      pointer destprime = dest;
      const_pointer src2prime = src2;
      for (size_t n = 0; n < dimensions[2]; n++)
      {
        VecDot(destprime, src1, src2prime, dimensions[1], strides1[1], strides2[0]);
        destprime += strides0[1];
        src2prime += strides2[1];
      }
      dest += strides0[0];
      src1 += strides1[0];
    }
  }

  inline void Scale(pointer dest, const_pointer src, const value_type factor,
                    std::span<const size_t,  1> dimensions,
                    std::span<const ssize_t, 1> strides0,
                    std::span<const ssize_t, 1> strides1) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      *dest = *src * factor;
      dest += strides0[0];
      src += strides1[0];
    }
  }

  template <size_t N>
  inline void Scale(pointer dest, const_pointer src, const value_type factor,
                    std::span<const size_t,  N> dimensions,
                    std::span<const ssize_t, N> strides0,
                    std::span<const ssize_t, N> strides1) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      Scale(dest, src, factor,
          std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides0.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1));
      dest += strides0[0];
      src += strides1[0];
    }
  }

 public:

  /// operator()() executes and returns a (scalar) tensor with the 'vector dot' multiplication.
  auto operator()() const requires (tensor1_rank == 1 && tensor2_rank == 1)
  {
    size_t dimensions = tensor1_.Dimensions()[0];
    auto result = Tensor(Uninitialized<value_type>{});

    VecDot(result.Data(),
           tensor1_.Data(),
           tensor2_.Data(),
           dimensions,
           tensor1_.Strides()[0],
           tensor2_.Strides()[0]);

    return result;
  }

  /// operator()() executes and returns a (matrix) tensor for a mtrix multiplication.
  auto operator()() const requires (tensor1_rank == 2 && tensor2_rank == 2)
  {
    auto&& dimensions1 = tensor1_.Dimensions();
    auto&& dimensions2 = tensor2_.Dimensions();
    size_t dimensions[] = {dimensions1[0], dimensions1[1], dimensions2[1]};
    auto result = Tensor({dimensions1[0], dimensions2[1]}, Uninitialized<value_type>{});

    MatMul(result.Data(),
           tensor1_.Data(),
           tensor2_.Data(),
           std::span(dimensions),
           std::span(result.Strides()),
           std::span(tensor1_.Strides()),
           std::span(tensor2_.Strides()));

    return result;
  }

  /// operator()() execute and returns a tensor of the same rank for a matrix/vector * scalar multiplication.
  auto operator()() const requires (tensor2_rank == 0)
  {
    auto&& dimensions = tensor1_.Dimensions();
    auto result = Tensor(dimensions, Uninitialized<value_type>{});

    Scale(result.Data(),
          tensor1_.Data(),
          *tensor2_.Data(),
          std::span(dimensions),
          std::span(result.Strides()),
          std::span(tensor1_.Strides()));
    return result;
  }

  /// operator()() execute and returns a tensor of the same rank for a scalar * matrix/vector multiplication.
  auto operator()() const requires (tensor1_rank == 0 && tensor2_rank != 0)
  {
    auto&& dimensions = tensor2_.Dimensions();
    auto result = Tensor(dimensions, Uninitialized<value_type>{});

    Scale(result.Data(),
          tensor2_.Data(),
          *tensor1_.Data(),
          std::span(dimensions),
          std::span(result.Strides()),
          std::span(tensor2_.Strides()));

    return result;
  }

  /// operator()() executes and returns a (vector) tensor of a matrix * vector multiplication.
  auto operator()() const requires (tensor1_rank == 2 && tensor2_rank == 1)
  {
    auto&& dimensions1 = tensor1_.Dimensions();
    auto result = Tensor(dimensions1[0], Uninitialized<value_type>{});
    size_t dimensions[] = {dimensions1[0], dimensions1[1], 1};

    std::array<ssize_t, 2> strides0 = {result.Strides()[0], 0};
    std::array<ssize_t, 2> strides2 = {tensor2_.Strides()[0], 0};

    // Use: M_m_n * V_n = M_m_n * V_n_1 -> V_m
    MatMul(result.Data(),
           tensor1_.Data(),
           tensor2_.Data(),
           std::span(dimensions),
           std::span(strides0),
           std::span(tensor1_.Strides()),
           std::span(strides2));

    return result;
  }

  /// operator()() executes and returns a (vector) tensor of a vector * matrix multiplication.
  auto operator()() const requires (tensor1_rank == 1 && tensor2_rank == 2)
  {
    auto&& dimensions2 = tensor2_.Dimensions();
    auto result = Tensor(dimensions2[1], Uninitialized<value_type>{});
    size_t dimensions[] = {1, dimensions2[0], dimensions2[1]};

    std::array<ssize_t, 2> strides  = {0L, result.Strides()[0]};
    std::array<ssize_t, 2> strides1 = {0L, tensor1_.Strides()[0]};

    // Use V_m * M_m_n = V_1_m * M_m_n -> V_n
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
  TTensor1 tensor1_;
  TTensor2 tensor2_;
};

//
// CTAD
//

namespace {
template <typename TTensor1, typename TTensor2>
struct matmul_rank
{
  constexpr static size_t tensor1_rank = std::remove_reference_t<TTensor1>::rank;
  constexpr static size_t tensor2_rank = std::remove_reference_t<TTensor2>::rank;
  constexpr static size_t value =
    (tensor1_rank == 0 ? tensor2_rank :
     tensor2_rank == 0 ? tensor1_rank :
     std::min(tensor1_rank, tensor2_rank));
};
}

template <ConvertibleTo<Tensor> TTensor1, ConvertibleTo<Tensor> TTensor2>
TensorMatMul(TTensor1&&, TTensor2&&)
  -> TensorMatMul<Tensor,
                  typename std::remove_reference_t<TTensor2>::value_type,
                  matmul_rank<TTensor1, TTensor2>::value,
                  typename to_tensor<TTensor1>::type,
                  typename to_tensor<TTensor2>::type>;

template <ConvertibleTo<Tensor> TTensor, Arithmetic T>
TensorMatMul(TTensor&&, T)
  -> TensorMatMul<Tensor,
                  typename std::remove_reference_t<TTensor>::value_type,
                  std::remove_reference_t<TTensor>::rank,
                  typename to_tensor<TTensor>::type,
                  Tensor<typename std::remove_cvref_t<TTensor>::value_type, 0>>;

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_MATMULH
