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

namespace grid {

template <typename _Tp, size_t>
inline std::enable_if_t<!std::is_floating_point_v<_Tp>, bool>
equals(const _Tp* src1, const _Tp* src2,
       std::span<const size_t,  0>,
       std::span<const ssize_t, 0>,
       std::span<const ssize_t, 0>)
{
  return *src1 == *src2;
}

template <typename _Tp, size_t>
inline std::enable_if_t<std::is_floating_point_v<_Tp>, bool>
equals(const _Tp* src1, const _Tp* src2,
       std::span<const size_t,  0>,
       std::span<const ssize_t, 0>,
       std::span<const ssize_t, 0>)
{
  auto max = std::max(std::abs(*src1), std::abs(*src2));
  _Tp eps = max * std::numeric_limits<_Tp>::epsilon();
  return std::abs(*src1 - *src2) <= eps;
}

template <typename _Tp, size_t>
inline std::enable_if_t<!std::is_floating_point_v<_Tp>, bool>
equals(const _Tp* src1, const _Tp* src2,
       std::span<const size_t,  1> dimensions,
       std::span<const ssize_t, 1> strides1,
       std::span<const ssize_t, 1> strides2)
{
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    if (*src1 != *src2)
      return false;
    reinterpret_cast<const char*&>(src1) += strides1[0];
    reinterpret_cast<const char*&>(src2) += strides2[0];
  }
  return true;
}

template <typename _Tp, size_t>
inline std::enable_if_t<std::is_floating_point_v<_Tp>, bool>
equals(const _Tp* src1, const _Tp* src2,
       std::span<const size_t,  1> dimensions,
       std::span<const ssize_t, 1> strides1,
       std::span<const ssize_t, 1> strides2)
{
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    auto max = std::max(std::abs(*src1), std::abs(*src2));
    _Tp eps = max * std::numeric_limits<_Tp>::epsilon();

    if (std::abs(*src1 - *src2) > eps)
      return false;

    reinterpret_cast<const char*&>(src1) += strides1[0];
    reinterpret_cast<const char*&>(src2) += strides2[0];
  }
  return true;
}

template <typename _Tp, size_t _N>
inline std::enable_if_t<(_N > 1), bool>
equals(const _Tp* src1, const _Tp* src2,
       std::span<const size_t,  _N> dimensions,
       std::span<const ssize_t, _N> strides1,
       std::span<const ssize_t, _N> strides2)
{
  static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    if (!equals<_Tp, _N - 1>(src1, src2,
                             std::span<const size_t,  _N - 1>(dimensions.begin() + 1, _N - 1),
                             std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1),
                             std::span<const ssize_t, _N - 1>(strides2.begin() + 1, _N - 1)))
      return false;

    reinterpret_cast<const char*&>(src1) += strides1[0];
    reinterpret_cast<const char*&>(src2) += strides2[0];
  }
  return true;
}

// TODO: will https://open-std.org/JTC1/SC22/WG21/docs/papers/2019/p1045r1.html help for using tensor.Rank() as constexpr?
template <PrimitiveTensor  _Tensor1, PrimitiveTensor _Tensor2>
bool operator==(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  constexpr size_t _Rank = std::remove_cvref_t<decltype(tensor1)>::rank;
  static_assert(_Rank == std::remove_cvref_t<decltype(tensor2)>::rank,
                "ranks mismatch between tensors");

  return equals<typename std::remove_cvref_t<_Tensor1>::value_type, _Rank>(
                      tensor1.Data(),
                      tensor2.Data(),
                      std::span(tensor1.Dimensions()),
                      std::span(tensor1.Strides()),
                      std::span(tensor2.Strides()));
}

} // end of namespace grid

#endif // GRID_TENSOR_BASE_COMPARISON_H
