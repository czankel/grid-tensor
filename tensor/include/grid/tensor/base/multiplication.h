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

#include <algorithm>

namespace grid {


/// TensorMul<Tensor> implements tensor multiplication operation for tensors of the same rank.
/// Including matrix multiplication (MatMul) and vector dot-product (VecDot).
template <typename _T, size_t _Rank, PrimitiveTensor _Tensor1, PrimitiveTensor _Tensor2>
struct TensorMul<Tensor, _T, _Rank, _Tensor1, _Tensor2>
{
  using value_type = _T;

  template <ConvertibleTo<Tensor> T1, ConvertibleTo<Tensor> T2>
  TensorMul(T1&& tensor1, T2&& tensor2)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {}

  ~TensorMul() {}

  // delete assignment and copy/move constructors
  TensorMul() = delete;
  TensorMul(const TensorMul& other) = delete;
  TensorMul(TensorMul&& other) = delete;
  TensorMul& operator=(const TensorMul& other) = delete;
  TensorMul& operator=(TensorMul&& other) = delete;

  inline void VecDot(char* dest, const char* src1, const char* src2,
                     std::span<const size_t,  1> dims,
                     std::span<const ssize_t, 1> strides1,
                     std::span<const ssize_t, 1> strides2) const
  {
    value_type sum{0};
    for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
      sum += *reinterpret_cast<const value_type*>(src1) * *reinterpret_cast<const value_type*>(src2);
    *reinterpret_cast<value_type*>(dest) = sum;
  }

  inline void MatMul(char* dest, const char* src1, const char* src2,
                     std::span<const size_t,  2> dims,
                     std::span<const ssize_t, 2> strides0,
                     std::span<const ssize_t, 2> strides1,
                     std::span<const ssize_t, 2> strides2) const
  {
    for (size_t m = 0; m < dims[0]; m++, dest += strides0[0], src1 += strides1[0])
    {
      char* destprime = dest;
      const char* src2prime = src2;
      for (size_t n = 0; n < dims[0]; n++, destprime += strides0[1], src2prime += strides0[1])
        VecDot(destprime, src1, src2prime,
               std::span<const size_t,  1>(dims.begin() + 1, dims.end()),
               std::span<const ssize_t, 1>(strides1.begin() + 1, strides1.end()),
               std::span<const ssize_t, 1>(strides2.begin() + 1, strides2.end()));
    }
  }

  inline void Scale(char* dest, const char* src, const value_type factor,
                    std::span<const size_t,  1> dims,
                    std::span<const ssize_t, 1> strides0,
                    std::span<const ssize_t, 1> strides1) const
  {
    for (size_t i = 0; i < dims[0]; i++, dest += strides0[0], src += strides1[0])
      *reinterpret_cast<value_type*>(dest) = *reinterpret_cast<const value_type*>(src) * factor;
  }

  template <size_t _N>
  inline void Scale(char* dest, const char* src, const value_type factor,
                    std::span<const size_t,  _N> dims,
                    std::span<const ssize_t, _N> strides0,
                    std::span<const ssize_t, _N> strides1) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dims[0]; i++, dest += strides0[0], src += strides1[0])
      Scale(dest, src, factor,
          std::span<const size_t,  _N - 1>(dims.begin() + 1, _N - 1),
          std::span<const ssize_t, _N - 1>(strides0.begin() + 1, _N - 1),
          std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1));
  }


  auto operator()() const requires (_Tensor1::rank == 1 && _Tensor2::rank == 1)
  {
    auto& dims = tensor1_.Dims();
    auto result = Tensor(Uninitialized<value_type>{});

    VecDot(reinterpret_cast<char*>(result.Data()),
        reinterpret_cast<const char*>(tensor1_.Data()),
        reinterpret_cast<const char*>(tensor2_.Data()),
        std::span(dims),
        std::span(tensor1_.Strides()),
        std::span(tensor2_.Strides()));

    return result;
  }

  auto operator()() const requires (_Tensor1::rank == 2 && _Tensor2::rank == 2)
  {
    auto& dims = tensor1_.Dims();
    auto result = Tensor({dims[0], dims[0]}, Uninitialized<value_type>{});

    // transpose second matrix
    auto strides = tensor2_.Strides();
    std::swap(strides[0], strides[1]);

    MatMul(reinterpret_cast<char*>(result.Data()),
        reinterpret_cast<const char*>(tensor1_.Data()),
        reinterpret_cast<const char*>(tensor2_.Data()),
        std::span(dims),
        std::span(result.Strides()),
        std::span(tensor1_.Strides()),
        std::span(strides));
    return result;
  }

  auto operator()() const requires (_Tensor2::rank == 0)
  {
    auto& dims = tensor1_.Dims();
    auto result = Tensor(dims, Uninitialized<value_type>{});
    Scale(reinterpret_cast<char*>(result.Data()),
          reinterpret_cast<const char*>(tensor1_.Data()),
          *reinterpret_cast<const value_type*>(tensor2_.Data()),
          std::span(dims),
          std::span(result.Strides()),
          std::span(tensor1_.Strides()));
    return result;
  }

  auto operator()() const requires (_Tensor1::rank == 0 && _Tensor2::rank != 0)
  {
    auto& dims = tensor2_.Dims();
    auto result = Tensor(dims, Uninitialized<value_type>{});
    Scale(reinterpret_cast<char*>(result.Data()),
          reinterpret_cast<const char*>(tensor2_.Data()),
          *reinterpret_cast<const value_type*>(tensor1_.Data()),
          std::span(dims),
          std::span(result.Strides()),
          std::span(tensor2_.Strides()));
    return result;
  }


  _Tensor1 tensor1_;
  _Tensor2 tensor2_;
};


// CTAD

template <ConvertibleTo<Tensor> _Tensor1, ConvertibleTo<Tensor> _Tensor2>
TensorMul(_Tensor1, _Tensor2)
  -> TensorMul<Tensor, typename _Tensor2::value_type, std::max(_Tensor1::rank, _Tensor2::rank),
               typename to_tensor<_Tensor1>::type, typename to_tensor<_Tensor2>::type>;


} // end of namespace grid

#endif  // GRID_TENSOR_BASE_MULTIPLY_H
