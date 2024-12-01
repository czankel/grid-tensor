//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_SLOWCPU_OPERATORS_H
#define GRID_TENSOR_SLOWCPU_OPERATORS_H

namespace grid {


namespace details {

template <typename _T, size_t>
inline std::enable_if_t<!std::is_floating_point_v<_T>, bool>
equals(const char* src1, const char* src2,
       std::span<const size_t,  1> dims,
       std::span<const ssize_t, 1> strides1,
       std::span<const ssize_t, 1> strides2)
{
  for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
    if (*reinterpret_cast<const _T*>(src1) != *reinterpret_cast<const _T*>(src2))
      return false;
  return true;
}

template <typename _T, size_t>
inline std::enable_if_t<std::is_floating_point_v<_T>, bool>
equals(const char* src1, const char* src2,
       std::span<const size_t,  1> dims,
       std::span<const ssize_t, 1> strides1,
       std::span<const ssize_t, 1> strides2)
{
  constexpr _T max_abs_error = std::numeric_limits<_T>::epsilon() * 100;

  for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
  {
    _T data0 = *reinterpret_cast<const _T*>(src1);
    _T data1 = *reinterpret_cast<const _T*>(src2);

    if (std::abs(data0 - data1) > max_abs_error)
      return false;
  }
  return true;
}

template <typename _T, size_t _N>
inline std::enable_if_t<(_N > 1), bool>
equals(const char* src1, const char* src2,
       std::span<const size_t,  _N> dims,
       std::span<const ssize_t, _N> strides1,
       std::span<const ssize_t, _N> strides2)
{
  static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
  for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
    if (!equals<_T, _N - 1>(src1, src2,
                            std::span<const size_t,  _N - 1>(dims.begin() + 1, _N - 1),
                            std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1),
                            std::span<const ssize_t, _N - 1>(strides2.begin() + 1, _N - 1)))
      return false;

  return true;
}

} // end of namespace details


template <TensorFor<TensorSlowCpu> _Tensor1, TensorFor<TensorSlowCpu> _Tensor2>
bool operator==(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  constexpr size_t _Rank = tensor1.Rank();
  static_assert(_Rank == tensor2.Rank(), "ranks mismatch between tensors");

  return details::equals<typename std::remove_cvref_t<_Tensor1>::value_type, _Rank>(
                         reinterpret_cast<const char*>(tensor1.Data()),
                         reinterpret_cast<const char*>(tensor2.Data()),
                         std::span(tensor1.Dims()),
                         std::span(tensor1.Strides()),
                         std::span(tensor2.Strides()));
}


/// TensorAdd<TensorSlowCpu> implements tensor addition operation for tensors of the same rank.
template <typename _T, size_t _Rank, TensorFor<TensorSlowCpu> _Tensor1, TensorFor<TensorSlowCpu> _Tensor2>
struct TensorAdd<TensorSlowCpu, _T, _Rank, _Tensor1, _Tensor2> : TensorBaseOp //<_Tensor1>
{
  constexpr static size_t Rank()                  { return _Rank; }
  using tensor_type = TensorSlowCpu<_T, _Rank>;
  using value_type = _T;

  template <ConvertibleTensorFor<TensorSlowCpu> T1, ConvertibleTensorFor<TensorSlowCpu> T2>
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
          std::span<const size_t,  _N - 1>(dims.begin() + 1, _N - 1), //dims.end()),
          std::span<const ssize_t, _N - 1>(strides0.begin() + 1, _N - 1), // strides0.end()),
          std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1), // strides1.end()),
          std::span<const ssize_t, _N - 1>(strides2.begin() + 1, _N - 1)); // strides2.end()));
  }

  // Functor
  // FIXME: must be same rank? static_assert?
  auto operator()() const
  {
    auto& dims = tensor1_.Dims();
    auto result = TensorSlowCpu(dims, Uninitialized<value_type>{});

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

template <ConvertibleTensorFor<TensorSlowCpu> _Tensor1, ConvertibleTensorFor<TensorSlowCpu> _Tensor2>
TensorAdd(_Tensor1, _Tensor2)
  -> TensorAdd<TensorSlowCpu, typename _Tensor2::value_type, _Tensor1::Rank(),
               typename _Tensor1::tensor_type, typename _Tensor2::tensor_type>;

} // end of namespace grid

#endif // GRID_TENSOR_SLOWCPU_OPERATORS_H
