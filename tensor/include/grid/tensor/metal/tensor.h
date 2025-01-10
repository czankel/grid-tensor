//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_METAL_TENSOR_H
#define GRID_TENSOR_METAL_TENSOR_H

#include "array.h"

// Metal -> Metal
template <AnyTensor T1, AnyTensor T2>
requires is_same_tensor_memory_v<T1, DeviceMemory<device::Metal>>
inline void Copy(T1& tensor1, const T2& tensor2)
{
  tensor1.Copy(tensor2);
}

// System -> Metal
template <AnyTensor T1, AnyTensor T2>
requires is_same_tensor_memory_v<T1, DeviceMemory<device::Metal>> &&
         is_same_tensor_memory_v<T2, DeviceMemory<device::Base>>
inline void Copy(T1& tensor1, const T2& tensor2)
{
  printf("METAL COPY\n");
#if 0
  // FIXME: if any or both are views, which dimensions to pick?
  auto dimensions = tensor1.Dimensions();
  auto strides1 = tensor1.Strides();
  auto strides2 = other.Strides();

#if 0
  if (get_block_size<value_type>(dimensions, strides1) > size_)
    throw std::runtime_error("invalid size");
#endif

  details::copy_unsafe(tensor1.Data(), tensor2.Data(),
                       std::span<const size_t, N>(dimensions.begin(), N),
                       std::span<const ssize_t, N>(strides1.begin(), N),
                       std::span<const ssize_t, N>(strides2.begin(), N));
#endif
}

// Metal -> System

#endif  // GRID_TENSOR_METAL_TENSOR_H
