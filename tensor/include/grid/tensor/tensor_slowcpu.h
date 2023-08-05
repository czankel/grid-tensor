//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_SLOWCPU_H
#define GRID_TENSOR_SLOWCPU_H

#include <initializer_list>
#include <type_traits>
#include <array>
#include <algorithm>

#include "tensor.h"

namespace grid {

/// TensorSlowCpu is an unoptimized tensor implementatoin for the CPU.
template <size_t, typename _T, auto...> struct TensorSlowCpu;


/// TensorSlowCpu<_Rank, _T> is a specialization of TensorSlowCpu for a dynamically allocated buffer.
template <size_t _Rank, typename _T>
struct TensorSlowCpu<_Rank, _T> : TensorBase<TensorSlowCpu, _Rank, _T>
{
  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer.
  explicit TensorSlowCpu(unsigned int dim, _T init) : dim_(dim), data_(new _T[dim])
  {
    for (unsigned int i = 0; i < dim; i++)
      data_[i] = init;
  }

  /// Destructor
  ~TensorSlowCpu()                                { delete[] data_; }

  /// Dim returns the dimension of the rank.
  unsigned int Dim(unsigned int index) const      { return dim_[index]; }

  /// Data returns a pointer to the data buffer.
  _T* Data() const                                { return data_; }


  unsigned int dim_[_Rank];
  _T* data_;
};

// CTAD rules

// Tensor(int, _T) -> Rank-1 tensor with a dynamically allocated buffer.
template <typename _T>
explicit TensorSlowCpu(int, _T) -> TensorSlowCpu<1, _T>;

} // end of namespace grid

#endif  // GRID_TENSOR_SLOWCPU_H
