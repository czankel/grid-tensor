//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_ADDITION_H
#define GRID_TENSOR_BASE_ADDITION_H

namespace grid {

/// TensorAdd<Tensor> implements tensor addition operation for tensors of the same rank.
template <typename _Tp, size_t _Rank, PrimitiveTensor _Tensor1, PrimitiveTensor _Tensor2>
requires (_Tensor1::Rank() == _Tensor2::Rank())
class TensorAdd<Tensor, _Tp, _Rank, _Tensor1, _Tensor2>
{
 public:
  using value_type = _Tp;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = _Rank;

  template <ConvertibleTo<Tensor> T1, ConvertibleTo<Tensor> T2>
  TensorAdd(T1&& tensor1, T2&& tensor2)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {}

  ~TensorAdd() {}

  // delete assignment and copy/move constructors
  TensorAdd() = delete;
  TensorAdd(const TensorAdd& other) = delete;
  TensorAdd(TensorAdd&& other) = delete;
  TensorAdd& operator=(const TensorAdd& other) = delete;
  TensorAdd& operator=(TensorAdd&& other) = delete;

 private:
  inline void add(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  0> dimensions,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    *dest = *src1 + *src2;
  }

  inline void add(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  1> dimensions,
                  std::span<const ssize_t, 1>,
                  std::span<const ssize_t, 1> strides1,
                  std::span<const ssize_t, 1> strides2) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      dest[i] = *src1 + *src2;
      reinterpret_cast<const char*&>(src1) += strides1[0];
      reinterpret_cast<const char*&>(src2) += strides2[0];
    }
  }

  template <size_t _N> inline
  void add(pointer dest, const_pointer src1, const_pointer src2,
           std::span<const size_t,  _N> dimensions,
           std::span<const ssize_t, _N> strides0,
           std::span<const ssize_t, _N> strides1,
           std::span<const ssize_t, _N> strides2) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      add(dest, src1, src2,
          std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
          std::span<const ssize_t, _N - 1>(strides0.begin() + 1, _N - 1),
          std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1),
          std::span<const ssize_t, _N - 1>(strides2.begin() + 1, _N - 1));

      reinterpret_cast<char*&>(dest) += strides0[0];
      reinterpret_cast<const char*&>(src1) += strides1[0];
      reinterpret_cast<const char*&>(src2) += strides2[0];
    }
  }

 public:

  /// operator()() executes the operation and returns a tensor.
  auto operator()() const
  {
    auto& dimensions = tensor1_.Dimensions();
    auto result = Tensor(dimensions, Uninitialized<value_type>{});

    add(result.Data(),
        tensor1_.Data(),
        tensor2_.Data(),
        std::span<const size_t, _Rank>(dimensions),
        std::span<const ssize_t, _Rank>(result.Strides()),
        std::span<const ssize_t, _Rank>(tensor1_.Strides()),
        std::span<const ssize_t, _Rank>(tensor2_.Strides()));
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
TensorAdd(_Tensor1, _Tensor2)
  -> TensorAdd<Tensor, typename _Tensor1::value_type, _Tensor1::rank,
               typename to_tensor<_Tensor1>::type, typename to_tensor<_Tensor2>::type>;

} // end of namespace grid

#endif // GRID_TENSOR_BASE_ADDITION_H
