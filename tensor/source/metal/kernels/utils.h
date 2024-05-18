//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef TENSOR_SOURCE_METAL_KERNELS_UTILS_H
#define TENSOR_SOURCE_METAL_KERNELS_UTILS_H

namespace metal {

inline size_t pos_to_index(uint pos, constant const size_t& stride) {
  return pos * stride;
}

inline size_t pos_to_index(uint2 pos, constant const size_t strides[2]) {
  return pos.x * strides[1] + pos.y * strides[0];
}

inline size_t pos_to_index(uint3 pos, constant const size_t strides[3]) {
  return pos.x * strides[2] + pos.y * strides[1] + pos.z * strides[0];
}

} // end of namespace metal

#endif  // TENSOR_SOURCE_METAL_KERNELS_UTILS_H
