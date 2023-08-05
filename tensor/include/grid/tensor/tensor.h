//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_H
#define GRID_TENSOR_TENSOR_H

#include <iostream>
#include <iomanip>

namespace grid {

/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename> struct Uninitialized {};


/// TensorBase provides a base class for derived "runtime" tensor implementations with
/// optimizations specific to CPUs and accelerators.
///
/// @tparam _Rank      Rank of the tensor with rank-0 being a scalar, 1 a vector, etc.
/// @tparam _TensorRT  Derived class with the actual implementation for a specific "runtime".
///                    This can be an optimized or unoptimized implementation for the CPU or an
///                    implementation for an accelerator.
/// @tparam _T         Underlying numeric type (int, float, etc.) of the tensor.
/// @tparam _Args      Additional non-type parameters for the specific TensorRT implementation.
template <template <size_t, typename, auto...> typename _TensorRT, size_t _Rank, typename _T, auto... _Args>
struct TensorBase
{
  using TensorType = _TensorRT<_Rank, _T, _Args...>;
  using Type = _T;

  // Rank returns the rank of the tensor.
  size_t Rank()                                   { return _Rank; }
};

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_H
