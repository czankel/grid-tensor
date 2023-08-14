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


// helper functions to identify if a Tensor is for a specific runtime
template <typename, template <size_t, typename, auto...> typename>
struct is_tensor_runtime : std::false_type {};

template <template <size_t, typename, auto...> typename _Tensor, size_t _Rank, typename _T, auto... _Args>
struct is_tensor_runtime<_Tensor<_Rank, _T, _Args...>, _Tensor> : std::true_type {};


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

  /// operator<< outputs the tensor buffer.
  inline friend std::ostream& operator<<(std::ostream& os, const TensorType& t)
  {
    std::function<void(int, const _T*&)> print;
    print = [&os, &t, &print](size_t index, const _T*& ptr) {
      os << "{ ";
      if (index < _Rank - 1)
      {
        for (size_t i = t.Dim(index); i > 0; i--)
        {
          print(index + 1, ptr);
          if (i != 1)
            os << ", ";
          else
            os << " }";
        }
      }
      else
      {
        for (size_t i = t.Dim(index); i > 0; i--)
        {
          os << *ptr++;
          if (i != 1)
            os << ", ";
          else
            os << " }";
        }
      }
    };
    const _T* ptr = t.Data();
    print(0, ptr);
    os << std::flush;

    return os;
  }

  // Rank returns the rank of the tensor.
  size_t Rank()                                   { return _Rank; }
};

// Concepts to define Tensors of a specific Rank.

template <typename _Tensor> concept TensorR0Type = _Tensor::_Rank == 0;
template <typename _Tensor> concept TensorR1Type = _Tensor::_Rank == 1;
template <typename _Tensor> concept TensorR2Type = _Tensor::_Rank == 2;
template <typename _Tensor> concept TensorR3Type = _Tensor::_Rank == 3;


} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_H
