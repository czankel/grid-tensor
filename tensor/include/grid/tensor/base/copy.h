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
template <typename T, size_t>
inline void
copy(T* dst, const T* src,
     std::span<const size_t,  0>,
     std::span<const ssize_t, 0>,
     std::span<const ssize_t, 0>)
{
  return *dst = *src;
}

template <typename T, size_t>
inline void
copy(T* dst, const T* src,
     std::span<const size_t,  1> dimensions,
     std::span<const ssize_t, 1> strides1,
     std::span<const ssize_t, 1> strides2)
{
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    *dst = *src;
    dst += strides1[0];
    src += strides2[0];
  }
}

template <typename T, size_t N>
inline std::enable_if_t<(N > 1), void>
copy(T* dst, const T* src,
       std::span<const size_t,  N> dimensions,
       std::span<const ssize_t, N> strides1,
       std::span<const ssize_t, N> strides2)
{
  static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    copy<T, N - 1>(dst, src,
                     std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
                     std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1),
                     std::span<const ssize_t, N - 1>(strides2.begin() + 1, N - 1));
    dst += strides1[0];
    src += strides2[0];
  }
}

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_COPY_H
