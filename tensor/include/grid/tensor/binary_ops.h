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

#include "concepts.h"

namespace grid {
namespace tensor {

/// BinaryOp<Tensor> implements element-wise binary operations of two tensors.
/// The dimensions of the tensors must match following broadcasting rules.
/// The resulting rank is the maximum of the tensor ranks.
///
///  @tparm T         resulting value type
///  @tparm TOp       binary operation
///  @tparm TTensor1  LHS tensor
///  @tparm TTensor2  RHS tensor
///
/// Broadcasting rules:
///
///  - Lower ranking tensors are extended and filled with dimension 1 and stride 0 on the left.
///  - Staring from the right, the dimensions must be either identical or 1.
///
/// Examples:
///   shape: 3, 4, 4 <op> shape: 1, 4, 1    -> OK
///   shape:    4, 1 <op> shape: 3, 4, 3    -> OK
///   shape: 3, 4, 4 <op> shape: 3, 5, 1    -> Error
///
template <typename T, typename TOp, PrimitiveTensor TTensor1, PrimitiveTensor TTensor2>
class BinaryOp
{
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using tensor1_type = std::remove_reference_t<TTensor1>;
  using tensor2_type = std::remove_reference_t<TTensor2>;
  constexpr static size_t rank = std::max(tensor1_type::rank, tensor2_type::rank);

  template <typename T1, typename T2>
  BinaryOp(T1&& tensor1, T2&& tensor2)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {}

  ~BinaryOp() {}

  // delete assignment and copy/move constructors
  BinaryOp() = delete;
  BinaryOp(const BinaryOp& other) = delete;
  BinaryOp(BinaryOp&& other) = delete;
  BinaryOp& operator=(const BinaryOp& other) = delete;
  BinaryOp& operator=(BinaryOp&& other) = delete;

 private:
  // operation on a single element
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  0> dimensions,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    TOp::eval(dest, src1, src2);
  }

  // operation on a single dimension (unoptimized)
  inline void eval(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  1> dimensions,
                  std::span<const ssize_t, 1>,
                  std::span<const ssize_t, 1> strides1,
                  std::span<const ssize_t, 1> strides2) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      TOp::eval(dest + i, src1, src2);
      reinterpret_cast<const char*&>(src1) += strides1[0];
      reinterpret_cast<const char*&>(src2) += strides2[0];
    }
  }

  // operation on higher dimensions (unoptimized)
  template <size_t N> inline
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

  /// operator()() evaluates the binary operation and returns a tensor.
  auto operator()() const
  {
    auto [dimensions, strides1, strides2] = Broadcast(tensor1_, tensor2_);
    auto result = Tensor(dimensions, Uninitialized<value_type>{});

    eval(result.Data(),
        tensor1_.Data(),
        tensor2_.Data(),
        std::span<const size_t, rank>(dimensions),
        std::span<const ssize_t, rank>(result.Strides()),
        std::span<const ssize_t, rank>(strides1),
        std::span<const ssize_t, rank>(strides2));
    return result;
  }

 private:
  TTensor1 tensor1_;
  TTensor2 tensor2_;
};

} // end of namespace tensor
} // end of namespace grid

#endif // GRID_TENSOR_BINARY_OP_H
