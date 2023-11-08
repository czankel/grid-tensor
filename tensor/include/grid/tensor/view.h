//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_COPY_H
#define GRID_TENSOR_BASE_COPY_H

#include <type_traits>

namespace grid {

namespace details {

template <typename _T, size_t>
inline void
copy(const char* src1, const char* src2,
     std::span<const size_t,  0>,
     std::span<const ssize_t, 0>,
     std::span<const ssize_t, 0>)
{
  return *reinterpret_cast<const _T*>(src1) = *reinterpret_cast<const _T*>(src2);
}

template <typename _T, size_t>
inline void
copy(const char* src1, const char* src2,
     std::span<const size_t,  1> dims,
     std::span<const ssize_t, 1> strides1,
     std::span<const ssize_t, 1> strides2)
{
  for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
    *reinterpret_cast<const _T*>(src1) = *reinterpret_cast<const _T*>(src2);
}

template <typename _T, size_t _N>
inline std::enable_if_t<(_N > 1), bool>
copy(const char* src1, const char* src2,
       std::span<const size_t,  _N> dims,
       std::span<const ssize_t, _N> strides1,
       std::span<const ssize_t, _N> strides2)
{
  static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
  for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
    !copy<_T, _N - 1>(src1, src2,
                      std::span<const size_t,  _N - 1>(dims.begin() + 1, _N - 1),
                      std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1),
                      std::span<const ssize_t, _N - 1>(strides2.begin() + 1, _N - 1));
}

} // end of namespace details

#ifndef GRID_TENSOR_BASE_COPY_H
