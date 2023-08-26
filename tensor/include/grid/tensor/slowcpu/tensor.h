//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_SLOWCPU_TENSOR_H
#define GRID_TENSOR_SLOWCPU_TENSOR_H

#include <initializer_list>
#include <type_traits>
#include <array>
#include <algorithm>
#include <numeric>

#include "tensor.h"

namespace grid {

/// TensorSlowCpu<1, _T, _N> is a specialization of a rank-1 tensor (vector) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _N>
struct TensorSlowCpu<1, _T, _N> : TensorBase
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
      stride_(_N),
      array_(get_array(std::move(init)))
  {}

  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                  { return 1UL; }

  /// Dim returns the dimension for the rank.
  unsigned int Dim(unsigned int index) const      { if (index > 1) throw std::out_of_range ("index");
                                                    return dim_[index]; }
  /// Dim returns the stride for the rank.
  unsigned int Stride(unsigned int index) const   { if (index > 1) throw std::out_of_range ("index");
                                                    return stride_[index]; }
  /// Data returns a pointer to the data buffer.
  const _T* Data() const                          { return array_.data(); }


  unsigned int dim_[1];
  unsigned int stride_[1];
  std::array<_T, _N> array_;
};


/// TensorSlowCpu<_T, _M, _N> is a specialization of a rank-2 tensor (matrix) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _M, size_t _N>
struct TensorSlowCpu<2, _T, _M, _N> : TensorBase
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
      stride_(_M, _N),
      array_(get_array(std::move(init)))
  {}

  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                  { return 2UL; }

  /// Dim returns the dimension of the rank.
  unsigned int Dim(unsigned int index) const      { if (index > 2) throw std::out_of_range ("index");
                                                    return dim_[index]; }
  /// Dim returns the stride of the rank.
  unsigned int Stride(unsigned int index) const   { if (index > 2) throw std::out_of_range ("index");
                                                    return stride_[index]; }
  /// Data returns a pointer to the data buffer.
  const _T* Data() const                          { return array_.data(); }


  unsigned int dim_[2];
  unsigned int stride_[2];
  std::array<_T, _M * _N> array_;
};


/// TensorSlowCpu<_Rank, _T> is a specialization of TensorSlowCpu for a dynamically allocated buffer.
template <size_t _Rank, typename _T>
struct TensorSlowCpu<_Rank, _T> : TensorBase
{
  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  explicit TensorSlowCpu(unsigned int dim, _T init) : dim_{dim}, stride_{dim}, data_(new _T[dim])
  {
    for (unsigned int i = 0; i < dim; i++)
      data_[i] = init;
  }

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(unsigned int dim, Uninitialized<_T>) : dim_{dim}, stride_{dim}, data_(new _T[dim]) {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit TensorSlowCpu(unsigned int dim_m, int dim_n, _T init)
    : dim_{dim_m, dim_n},
      stride_{dim_m, dim_n},
      data_(new _T[dim_m * dim_n])
  {
    for (unsigned int i = 0; i < dim_m * dim_n; i++)
      data_[i] = init;
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(unsigned int dim_m, int dim_n, Uninitialized<_T>)
  : dim_{dim_m, dim_n},
    stride_{dim_m, dim_n},
    data_(new _T[dim_m * dim_n])
  {}

  // helper function to extra brace-initializer list
  constexpr std::array<unsigned int, _Rank> get_array(std::initializer_list<unsigned int>&& dim)
  {
    std::array<unsigned int, _Rank> res;
    std::copy(dim.begin(), dim.end(), res.begin());
    return res;
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(std::initializer_list<unsigned int>&& dims, _T init)
    : dim_(get_array(std::move(dims))),
      stride_(dim_),
      data_(new _T[std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<unsigned int>())])
  {
    unsigned int count = std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<unsigned int>());
    for (unsigned int i = 0; i < count; i++)
      data_[i] = init;
  }


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(std::initializer_list<unsigned int>&& dims, Uninitialized<_T>)
    : dim_(get_array(std::move(dims))),
      stride_(dim_),
      data_(new _T[std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<unsigned int>())])
  { }

  /// ....
  explicit TensorSlowCpu(std::initializer_list<unsigned int>&& dims,
                         std::initializer_list<unsigned int>&& strides,
                         _T init)
    : dim_(get_array(std::move(dims))),
      stride_(get_array(std::move(strides))),
      data_(new _T[std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<unsigned int>())])
  {
    // TODO handle strides
    unsigned int count = std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<unsigned int>());
    for (unsigned int i = 0; i < count; i++)
      data_[i] = init;
  }

  explicit TensorSlowCpu(std::initializer_list<unsigned int>&& dims,
                         std::initializer_list<unsigned int>&& strides,
                         Uninitialized<_T>)
    : dim_(get_array(std::move(dims))),
      stride_(get_array(std::move(strides))),
      data_(new _T[std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<unsigned int>())])
  {
  }

  /// Destructor
  ~TensorSlowCpu()                                { delete[] data_; }

  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                  { return _Rank; }

  /// Dim returns the dimension of the rank.
  unsigned int Dim(unsigned int index) const      { return dim_[index]; }

  // TODOo: assert on the index; also Dim()
  /// Stride returns the stride of the rank.
  unsigned int Stride(unsigned int index) const   { return stride_[index]; }

  /// Data returns a pointer to the data buffer.
  _T* Data() const                                { return data_; }


  std::array<unsigned int, _Rank> dim_;
  std::array<unsigned int, _Rank> stride_;
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

// Tensor ...with stride
template <typename _T, size_t _N>
explicit TensorSlowCpu(unsigned int(&&d)[_N], unsigned int(&&s)[_N], _T) -> TensorSlowCpu<_N, _T>;

template <typename _T, size_t _N>
explicit TensorSlowCpu(unsigned int(&&d)[_N], unsigned int(&&s)[_N], Uninitialized<_T>) -> TensorSlowCpu<_N, _T>;


// Concepts for SlowCPU Tensors of different ranks.

template <typename _Tensor>
concept TensorSlowCpuR1Type = is_same_runtime_v<_Tensor, TensorSlowCpu> && _Tensor::Rank() == 1;
template <typename _Tensor>
concept TensorSlowCpuR2Type = is_same_runtime_v<_Tensor, TensorSlowCpu> && _Tensor::Rank() == 2;
template <typename _Tensor>
concept TensorSlowCpuR3Type = is_same_runtime_v<_Tensor, TensorSlowCpu> && _Tensor::Rank() == 3;
template <typename _Tensor>
concept TensorSlowCpuR4Type = is_same_runtime_v<_Tensor, TensorSlowCpu> && _Tensor::Rank() == 4;

} // end of namespace grid

#endif  // GRID_TENSOR_SLOWCPU_TENSOR_H
