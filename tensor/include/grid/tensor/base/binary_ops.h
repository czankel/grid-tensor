//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_BINARY_OPS_H
#define GRID_TENSOR_BASE_BINARY_OPS_H

namespace grid {
namespace tensor {

template<typename T> struct AddOp
{
  // scalar X scalar
  static inline void eval(T* dest, const T* src1, const T* src2) { *dest = *src1 + *src2; }
};

/// Add adds two tensors element-wise.
template <PrimitiveTensor T1, PrimitiveTensor T2>
auto Add(T1&& tensor1, T2&& tensor2)
{
  using T = typename std::decay_t<T1>::value_type;
  return BinaryOp<T, AddOp<T>, typename to_tensor<T1>::type, typename to_tensor<T2>::type>(
      std::forward<T1>(tensor1), std::forward<T2>(tensor2));
};

template<typename T> struct MulOp
{
  // scalar X scalar
  static inline void eval(T* dest, const T* src1, const T* src2) { *dest = *src1 * *src2; }
};

/// Mul multiplies two tensors element-wise.
template <PrimitiveTensor T1, PrimitiveTensor T2>
auto Mul(T1&& tensor1, T2&& tensor2)
{
  using T = typename std::decay_t<T1>::value_type;
  return BinaryOp<T, MulOp<T>, typename to_tensor<T1>::type, typename to_tensor<T2>::type>(
      std::forward<T1>(tensor1), std::forward<T2>(tensor2));
};

} // end of namespace tensor
} // end of namespace grid

#endif  // GRID_TENSOR_BASE_BINARY_OPS_H
