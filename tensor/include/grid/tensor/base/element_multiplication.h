//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_ELEM_MULTIPLICATION_H
#define GRID_TENSOR_BASE_ELEM_MULTIPLICATION_H

namespace grid {

/// TensorElemMul<Tensor> implements an element-wise tensor multiplication.
/// This is different from matrix multiplication (MatMul) or the vector dot-product (VecDot).
template <typename _Tp, size_t _Rank, PrimitiveTensor _Tensor1, PrimitiveTensor _Tensor2>
class TensorElemMul<Tensor, _Tp, _Rank, _Tensor1, _Tensor2>
{
 public:
  using value_type = _Tp;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = _Rank;

  template <ConvertibleTo<Tensor> T1, ConvertibleTo<Tensor> T2>
  TensorElemMul(T1&& tensor1, T2&& tensor2)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {}

  template <ConvertibleTo<Tensor> T1, Scalar S>
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
  inline void ElemMul(pointer dest, const_pointer src1, const_pointer src2,
                      std::span<const size_t,  1> dimensions,
                      std::span<const ssize_t, 1> strides1,
                      std::span<const ssize_t, 1> strides2) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      dest[i] = *src1 * *src2;
      reinterpret_cast<const char*&>(src1) += strides1[0];
      reinterpret_cast<const char*&>(src2) += strides2[0];
    }
  }

  inline void ElemMul(pointer dest, const_pointer src1, const_pointer src2,
                      std::span<const size_t,  2> dimensions,
                      std::span<const ssize_t, 2> strides0,
                      std::span<const ssize_t, 2> strides1,
                      std::span<const ssize_t, 2> strides2) const
  {
    for (size_t m = 0; m < dimensions[0]; m++)
    {
      ElemMul(dest, src1, src2,
              std::span<const size_t,  1>(dimensions.begin() + 1, dimensions.end()),
              std::span<const ssize_t, 1>(strides1.begin() + 1, strides1.end()),
              std::span<const ssize_t, 1>(strides2.begin() + 1, strides2.end()));

      reinterpret_cast<char*&>(dest) += strides0[0];
      reinterpret_cast<const char*&>(src1) += strides1[0];
      reinterpret_cast<const char*&>(src2) += strides2[0];
    }
  }

 public:

  auto operator()() const
  {
    static_assert(_Tensor1::rank == _Tensor2::rank, "ranks must match for element-wise multiplication");
  }

  /// operator()() executes the operation and returns a tensor.
  auto operator()() const requires (_Tensor1::rank == _Tensor2::rank)
  {
#if 0
    auto dimensions1 = tensor1_.Dimensions();
    auto dimensions2 = tensor2_.Dimensions();

    std::array<size_t, rank> dimensions;
    for (size_t i = 2; i < _Rank; i++)
      dimensions[i] = dimensions1[i] == 1 ? dimensions2[i] : dimensions1[i];  // broadcast

    if (_Rank > 1)
    {
      dimensions[_Rank-2] = dimensions1[_Rank-2];
      dimensions[_Rank-1] = dimensions2[_Rank-1];
    }
    else
      dimensions[0] = dimensions1[0]; // FIXME: throw if different (or broadcast?)
#endif

    auto& dimensions = tensor1_.Dimensions();
    auto result = Tensor(dimensions, Uninitialized<value_type>{});

    ElemMul(result.Data(),
            tensor1_.Data(),
            tensor2_.Data(),
            std::span(dimensions),
            std::span(tensor1_.Strides()),
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
TensorElemMul(_Tensor1, _Tensor2)
  -> TensorElemMul<Tensor, typename _Tensor2::value_type, std::max(_Tensor1::rank, _Tensor2::rank),
               typename to_tensor<_Tensor1>::type, typename to_tensor<_Tensor2>::type>;

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_ELEM_MULTIPLICATION_H
