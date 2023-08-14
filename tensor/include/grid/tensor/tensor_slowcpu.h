//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_SLOWCPU_H
#define GRID_TENSOR_TENSOR_SLOWCPU_H

#include <initializer_list>
#include <type_traits>
#include <array>
#include <algorithm>

#include "tensor.h"

namespace grid {

/// TensorSlowCpu is an unoptimized tensor implementatoin for the CPU.
template <size_t, typename _T, auto...> struct TensorSlowCpu;


/// TensorSlowCpu<1, _T, _N> is a specialization of a rank-1 tensor (vector) for a 'static' array.
template <typename _T, size_t _N>
struct TensorSlowCpu<1, _T, _N> : TensorBase<TensorSlowCpu, 1, _T, _N>
{
  // helper function to initialize the std:array from an initializer list
  constexpr std::array<_T, _N> get_array(std::initializer_list<_T>&& init)
  {
    std::array<_T,_N> res;
    std::copy(init.begin(), init.end(), res.begin());
    return res;
  }

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit TensorSlowCpu(std::initializer_list<_T>&& init)
    : dim_(_N),
      array_(get_array(std::move(init)))
  {}

  /// Dim returns the dimension of the rank.
  unsigned int Dim(unsigned int index) const      { if (index > 1) throw std::out_of_range ("index");
                                                    return dim_[index]; }
  /// Data returns a pointer to the data buffer.
  const _T* Data() const                          { return array_.data(); }


  unsigned int dim_[1];
  std::array<_T, _N> array_;
};


/// TensorSlowCpu<_T, _M, _N> is a specialization of a rank-2 tensor (matrix) for a 'static' array.
template <typename _T, size_t _M, size_t _N>
struct TensorSlowCpu<2, _T, _M, _N> : TensorBase<TensorSlowCpu, 2, _T, _M, _N>
{
  // helper function to initialize the std:array from an initializer list
  constexpr std::array<_T, _M * _N> get_array(std::initializer_list<std::initializer_list<_T>>&& init)
  {
    std::array<_T, _M * _N> res{};
    auto line_it = res.begin();
    for (auto it : init)
    {
      std::copy(it.begin(), it.end(), line_it);
      line_it += _N;
    }
    return res;
  }

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit TensorSlowCpu(std::initializer_list<std::initializer_list<_T>>&& init)
    : dim_(_M, _N),
      array_(get_array(std::move(init)))
  {}

  /// Dim returns the dimension of the rank.
  unsigned int Dim(unsigned int index) const      { if  (index > 2) throw std::out_of_range ("index");
                                                    return dim_[index]; }
  /// Data returns a pointer to the data buffer.
  const _T* Data() const                          { return array_.data(); }


  unsigned int dim_[2];
  std::array<_T, _M * _N> array_;
};


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

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(unsigned int dim, Uninitialized<_T>) : dim_(dim), data_(new _T[dim]) {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer.
  explicit TensorSlowCpu(unsigned int dim_m, int dim_n, _T init)
    : dim_{dim_m, dim_n},
      data_(new _T[dim_m * dim_n])
  {
    for (unsigned int i = 0; i < dim_m * dim_n; i++)
      data_[i] = init;
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(unsigned int dim_m, int dim_n, Uninitialized<_T>)
  : dim_{dim_m, dim_n},
    data_(new _T[dim_m * dim_n])
  {}


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

// Tensor{Ts...} -> Rank-1 tensor with a static/local array (brace-initializer).
template <typename _T, typename... _Ts>
explicit TensorSlowCpu(_T, _Ts...) -> TensorSlowCpu<1, std::common_type_t<_T, _Ts...>, sizeof...(_Ts)+1>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <typename _T, size_t... _N>
TensorSlowCpu(_T(&&... l)[_N]) -> TensorSlowCpu<2, _T, sizeof...(_N), std::max({_N...})>;


// Tensor(uint, _T) -> Rank-1 tensor with a dynamically allocated buffer.
template <typename _T>
explicit TensorSlowCpu(unsigned int, _T) -> TensorSlowCpu<1, _T>;

// Tensor(uint, Uninitialized<T>) -> Rank-1 tensor with a dynamically allocated uninitialized buffer.
template <typename _T>
explicit TensorSlowCpu(unsigned int, Uninitialized<_T>) -> TensorSlowCpu<1, _T>;

// Tensor(uint, uint, _T) -> Rank-2 tensor with a dynamically allocated buffer.
template <typename _T>
explicit TensorSlowCpu(unsigned int, unsigned int, _T) -> TensorSlowCpu<2, _T>;

// Tensor(uint, Uninitialized<T>) -> Rank-2 tensor with a dynamically allocated uninitialized buffer.
template <typename _T>
explicit TensorSlowCpu(unsigned int, unsigned int, Uninitialized<_T>) -> TensorSlowCpu<2, _T>;

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_SLOWCPU_H
