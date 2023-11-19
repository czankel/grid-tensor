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
template <typename _T, size_t _Rank, PrimitiveTensor _Tensor1, PrimitiveTensor _Tensor2>
requires (_Tensor1::Rank() == _Tensor2::Rank())
struct TensorAdd<Tensor, _T, _Rank, _Tensor1, _Tensor2>
{
  using value_type = _T;
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

  inline void add(char* dest, const char* src1, const char* src2,
                  std::span<const size_t,  0> dims,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    *reinterpret_cast<value_type*>(dest) =
      *reinterpret_cast<const value_type*>(src1) + *reinterpret_cast<const value_type*>(src2);
  }


  // TODO: move conditional up the call chain (create addslow and addfast call-chains)
  inline void add(char* dest, const char* src1, const char* src2,
                  std::span<const size_t,  1> dims,
                  std::span<const ssize_t, 1>,
                  std::span<const ssize_t, 1> strides1,
                  std::span<const ssize_t, 1> strides2) const
  {
    value_type* datadest = reinterpret_cast<value_type*>(dest);
    for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
      datadest[i] = *reinterpret_cast<const value_type*>(src1) + *reinterpret_cast<const value_type*>(src2);
  }

  template <size_t _N> inline
  void add(char* dest, const char* src1, const char* src2,
           std::span<const size_t,  _N> dims,
           std::span<const ssize_t, _N> strides0,
           std::span<const ssize_t, _N> strides1,
           std::span<const ssize_t, _N> strides2) const
  {
    static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dims[0]; i++, dest += strides0[0], src1 += strides1[0], src2 += strides2[0])
      add(dest, src1, src2,
          std::span<const size_t,  _N - 1>(dims.begin() + 1, _N - 1),
          std::span<const ssize_t, _N - 1>(strides0.begin() + 1, _N - 1),
          std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1),
          std::span<const ssize_t, _N - 1>(strides2.begin() + 1, _N - 1));
  }

  // Functor
  // FIXME: must be same rank? static_assert?
  auto operator()() const
  {
    auto& dims = tensor1_.Dims();
    auto result = Tensor(dims, Uninitialized<value_type>{});

    add(reinterpret_cast<char*>(result.Data()),
        reinterpret_cast<const char*>(tensor1_.Data()),
        reinterpret_cast<const char*>(tensor2_.Data()),
        std::span(dims),
        std::span(result.Strides()),
        std::span(tensor1_.Strides()),
        std::span(tensor2_.Strides()));
    return result;
  }

  _Tensor1 tensor1_;
  _Tensor2 tensor2_;
};


// CTAD

template <ConvertibleTo<Tensor> _Tensor1, ConvertibleTo<Tensor> _Tensor2>
TensorAdd(_Tensor1, _Tensor2)
  -> TensorAdd<Tensor, typename _Tensor1::value_type, _Tensor1::rank,
               typename to_tensor<_Tensor1>::type, typename to_tensor<_Tensor2>::type>;

} // end of namespace grid

#endif // GRID_TENSOR_BASE_ADDITION_H
