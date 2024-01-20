//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BINARY_OP_H
#define GRID_TENSOR_BINARY_OP_H

#include <span>
#include <algorithm>
#include <ranges>

#include "../concepts.h"
#include "../binary_function.h"

namespace grid {
namespace tensor {

/// BinaryOperator<Operator> implements element-wise binary operations of two tensors.
/// The dimensions of the tensors must match following broadcasting rules.
/// The resulting rank is the maximum of the tensor ranks.
///
///  @tparm TOperator binary operator
template <typename TOperator>
class BinaryOperator
{
 private:
  // operation on a single element
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  0> dimensions,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    TOperator::eval(dest, src1, src2);
  }

  // operation on a single dimension (unoptimized)
  template <typename const_pointer, typename pointer>
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  1> dimensions,
                  std::span<const ssize_t, 1>,
                  std::span<const ssize_t, 1> strides1,
                  std::span<const ssize_t, 1> strides2) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      TOperator::eval(dest + i, src1, src2);
      reinterpret_cast<const char*&>(src1) += strides1[0];
      reinterpret_cast<const char*&>(src2) += strides2[0];
    }
  }

  // operation on higher dimensions (unoptimized)
  template <size_t N, typename const_pointer, typename pointer> inline
  //template <size_t N> inline
  void eval(pointer dest, const_pointer src1, const_pointer src2,
           std::span<const size_t,  N> dimensions,
           std::span<const ssize_t, N> strides0,
           std::span<const ssize_t, N> strides1,
           std::span<const ssize_t, N> strides2) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      eval(dest, src1, src2,
          std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides0.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides2.begin() + 1, N - 1));

      reinterpret_cast<char*&>(dest) += strides0[0];
      reinterpret_cast<const char*&>(src1) += strides1[0];
      reinterpret_cast<const char*&>(src2) += strides2[0];
    }
  }

 public:
  template <typename T, size_t TRank>
  void operator()(T* dst, const T* src1, const T* src2,
                  const std::array<size_t, TRank>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1,
                  const std::array<ssize_t, TRank>& strides2)
  {
    eval(dst, src1, src2,
      std::span<const size_t, TRank>(dimensions),
      std::span<const ssize_t, TRank>(strides0),
      std::span<const ssize_t, TRank>(strides1),
      std::span<const ssize_t, TRank>(strides2));
  }
};

//
// Operators
//

struct AddOperator
{
  // scalar X scalar
  template<typename T>
  static inline void eval(T* dest, const T* src1, const T* src2) { *dest = *src1 + *src2; }
};

struct SubOperator
{
  // scalar X scalar
  template<typename T>
  static inline void eval(T* dest, const T* src1, const T* src2) { *dest = *src1 - *src2; }
};

struct MulOperator
{
  // scalar X scalar
  template<typename T>
  static inline void eval(T* dest, const T* src1, const T* src2) { *dest = *src1 * *src2; }
};

struct DivOperator
{
  // scalar X scalar
  template<typename T>
  static inline void eval(T* dest, const T* src1, const T* src2) { *dest = *src1 / *src2; }
};

//
// Exported binary functions
//

/// @brief Add adds two tensors element-wise (lazily).
template <PrimitiveTensor TTensor1, PrimitiveTensor TTensor2>
auto Add(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return BinaryFunction(BinaryOperator<AddOperator>{}, std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

/// @brief Sub subtracts two tensors element-wise (lazily).
template <PrimitiveTensor TTensor1, PrimitiveTensor TTensor2>
auto Sub(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return BinaryFunction(SubOperator{}, std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

/// @brief Mul multiplies two tensors element-wise (lazily).
template <PrimitiveTensor TTensor1, PrimitiveTensor TTensor2>
auto Mul(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return BinaryFunction(MulOperator{}, std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

/// @brief Div multiplies two tensors element-wise (lazily).
template <PrimitiveTensor TTensor1, PrimitiveTensor TTensor2>
auto Div(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return BinaryFunction(DivOperator{}, std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}


} // end of namespace tensor
} // end of namespace grid

#endif // GRID_TENSOR_BINARY_OP_H
