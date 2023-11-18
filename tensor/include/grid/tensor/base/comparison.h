//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_COMPARISON_H
#define GRID_TENSOR_BASE_COMPARISON_H

#include <type_traits>

namespace grid {
namespace base {

template <typename _T, size_t>
inline std::enable_if_t<!std::is_floating_point_v<_T>, bool>
equals(const char* src1, const char* src2,
       std::span<const size_t,  0>,
       std::span<const ssize_t, 0>,
       std::span<const ssize_t, 0>)
{
  return *reinterpret_cast<const _T*>(src1) == *reinterpret_cast<const _T*>(src2);
}

template <typename _T, size_t>
inline std::enable_if_t<std::is_floating_point_v<_T>, bool>
equals(const char* src1, const char* src2,
       std::span<const size_t,  0>,
       std::span<const ssize_t, 0>,
       std::span<const ssize_t, 0>)
{
  constexpr _T max_abs_error = std::numeric_limits<_T>::epsilon() * 100;
  _T data0 = *reinterpret_cast<const _T*>(src1);
  _T data1 = *reinterpret_cast<const _T*>(src2);
  return std::abs(data0 - data1) <= max_abs_error;
}

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
  constexpr _T max_abs_error = std::numeric_limits<_T>::epsilon();

  for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
  {
    _T data0 = *reinterpret_cast<const _T*>(src1);
    _T data1 = *reinterpret_cast<const _T*>(src2);

    auto max = std::max(std::abs(data0), std::abs(data1));
    if (std::abs(data0 - data1) > max * max_abs_error)
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

} // end of namespace base


// TODO: will https://open-std.org/JTC1/SC22/WG21/docs/papers/2019/p1045r1.html help for using tensor.Rank() as constexpr?
template <TensorFor<base::Tensor> _Tensor1, TensorFor<base::Tensor> _Tensor2>
bool operator==(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  constexpr size_t _Rank = std::decay_t<decltype(tensor1)>::Rank();
  static_assert(_Rank == std::decay_t<decltype(tensor2)>::Rank(),
                "ranks mismatch between tensors");

  return base::equals<typename std::remove_cvref_t<_Tensor1>::value_type, _Rank>(
                      reinterpret_cast<const char*>(tensor1.Data()),
                       reinterpret_cast<const char*>(tensor2.Data()),
                       std::span(tensor1.Dims()),
                       std::span(tensor1.Strides()),
                       std::span(tensor2.Strides()));
}

} // end of namespace grid

#endif // GRID_TENSOR_BASE_COMPARISON_H
