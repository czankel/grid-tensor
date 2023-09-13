//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_SLOWCPU_MULTIPLY_H
#define GRID_TENSOR_SLOWCPU_MULTIPLY_H

namespace grid {


/// TensorMul<TensorSlowCpu> implements tensor multiplication operation for tensors of the same rank.
/// Including matrix multiplication (matmul) and vector dot-product (vecdot).
template <typename _T, size_t _Rank, TensorFor<TensorSlowCpu> _Tensor1, TensorFor<TensorSlowCpu> _Tensor2>
struct TensorMul<TensorSlowCpu, _T, _Rank, _Tensor1, _Tensor2> : TensorBaseOp
{
  constexpr static size_t Rank()                          { return _Rank; }
  using tensor_type = TensorSlowCpu<_T, _Rank>;
  using value_type = _T;

  template <ConvertibleTensorFor<TensorSlowCpu> T1, ConvertibleTensorFor<TensorSlowCpu> T2>
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

  inline void vecdot(char* dest, const char* src1, const char* src2,
                     std::span<const size_t,  1> dims,
                     std::span<const ssize_t, 1> strides1,
                     std::span<const ssize_t, 1> strides2) const
  {
    value_type sum{0};
    for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
      sum += *reinterpret_cast<const value_type*>(src1) * *reinterpret_cast<const value_type*>(src2);
    *reinterpret_cast<value_type*>(dest) = sum;
  }

  inline void matmul(char* dest, const char* src1, const char* src2,
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
        vecdot(destprime, src1, src2prime,
               std::span<const size_t,  1>(dims.begin() + 1, dims.end()),
               std::span<const ssize_t, 1>(strides1.begin() + 1, strides1.end()),
               std::span<const ssize_t, 1>(strides2.begin() + 1, strides2.end()));
    }
  }


  // Functors

  template <TensorRank<1> = _Tensor1, TensorRank<1> = _Tensor2>
  auto operator()() const
  {
    auto& dims = tensor1_.Dims();
    auto result = TensorSlowCpu(Uninitialized<value_type>{});

    vecdot(reinterpret_cast<char*>(result.Data()),
        reinterpret_cast<const char*>(tensor1_.Data()),
        reinterpret_cast<const char*>(tensor2_.Data()),
        std::span(dims),
        std::span(tensor1_.Strides()),
        std::span(tensor2_.Strides()));

    return result;
   }

  template <TensorRank<2> = _Tensor1, TensorRank<2> = _Tensor2>
  auto operator()() const
  {
    auto& dims = tensor1_.Dims();
    auto result = TensorSlowCpu({dims[0], dims[0]}, Uninitialized<value_type>{});

    // transpose second matrix
    auto strides = tensor2_.Strides();
    std::swap(strides[0], strides[1]);

    matmul(reinterpret_cast<char*>(result.Data()),
        reinterpret_cast<const char*>(tensor1_.Data()),
        reinterpret_cast<const char*>(tensor2_.Data()),
        std::span(dims),
        std::span(result.Strides()),
        std::span(tensor1_.Strides()),
        std::span(strides));
    return result;
  }

  _Tensor1 tensor1_;
  _Tensor2 tensor2_;
};


// CTAD

template <ConvertibleTensorFor<TensorSlowCpu> _Tensor1, ConvertibleTensorFor<TensorSlowCpu> _Tensor2>
TensorMul(_Tensor1, _Tensor2)
  -> TensorMul<TensorSlowCpu, typename _Tensor2::value_type, _Tensor1::Rank(),
               typename _Tensor1::tensor_type, typename _Tensor2::tensor_type>;


} // end of namespace grid

#endif  // GRID_TENSOR_SLOWCPU_MULTIPLY_H
