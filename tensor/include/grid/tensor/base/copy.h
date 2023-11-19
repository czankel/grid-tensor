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

#include <span>

namespace grid {

// copy copies the data between buffers accordig to dimensions and strides.
template <typename _Tp, size_t>
inline void
copy(_Tp* dst, const _Tp* src,
     std::span<const size_t,  0>,
     std::span<const ssize_t, 0>,
     std::span<const ssize_t, 0>)
{
  return *dst = *src;
}

template <typename _Tp, size_t>
inline void
copy(_Tp* dst, const _Tp* src,
     std::span<const size_t,  1> dims,
     std::span<const ssize_t, 1> strides1,
     std::span<const ssize_t, 1> strides2)
{
  for (size_t i = 0; i < dims[0]; i++)
  {
    *dst = *src;
    reinterpret_cast<char*&>(dst) += strides1[0];
    reinterpret_cast<const char*&>(src) += strides2[0];
  }
}

template <typename _Tp, size_t _N>
inline std::enable_if_t<(_N > 1), void>
copy(_Tp* dst, const _Tp* src,
       std::span<const size_t,  _N> dims,
       std::span<const ssize_t, _N> strides1,
       std::span<const ssize_t, _N> strides2)
{
  static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
  for (size_t i = 0; i < dims[0]; i++)
  {
    copy<_Tp, _N - 1>(dst, src,
                     std::span<const size_t,  _N - 1>(dims.begin() + 1, _N - 1),
                     std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1),
                     std::span<const ssize_t, _N - 1>(strides2.begin() + 1, _N - 1));
    reinterpret_cast<char*&>(dst) += strides1[0];
    reinterpret_cast<const char*&>(src) += strides2[0];
  }
}

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_COPY_H
