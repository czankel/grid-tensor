//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_SLOWCPU_TENSOR_H
#define GRID_TENSOR_SLOWCPU_TENSOR_H

#include <algorithm>
#include <array>
#include <initializer_list>
#include <numeric>

namespace grid {

/// TensorSlowCpu<_Rank, _T> is a specialization of TensorSlowCpu for a dynamically allocated buffer.
template <size_t _Rank, typename _T>
struct TensorSlowCpu<_Rank, _T> : TensorBase
{
  using value_type = _T;

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  // TODO: initializes entire data buffer based on stride dimensions
  explicit TensorSlowCpu(unsigned int dim, value_type init)
    : dim_{dim}, stride_{dim}, data_(new value_type[dim])
  {
    unsigned int count = std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<unsigned int>());
    for (unsigned int i = 0; i < count; i++)
      data_[i] = init;
  }

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(unsigned int dim, Uninitialized<value_type>)
    : dim_{dim}, stride_{dim}, data_(new value_type[dim])
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit TensorSlowCpu(unsigned int dim_m, int dim_n, value_type init)
    : dim_{dim_m, dim_n},
      stride_{dim_m, dim_n},
      data_(new value_type[dim_m * dim_n])
  {
    unsigned int count = std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<unsigned int>());
    for (unsigned int i = 0; i < count; i++)
      data_[i] = init;
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(unsigned int dim_m, int dim_n, Uninitialized<value_type>)
  : dim_{dim_m, dim_n},
    stride_{dim_m, dim_n},
    data_(new value_type[dim_m * dim_n])
  {}

  // helper function to extra brace-initializer list
  constexpr std::array<unsigned int, _Rank> get_array(std::initializer_list<unsigned int>&& dim)
  {
    std::array<unsigned int, _Rank> res;
    std::copy(dim.begin(), dim.end(), res.begin());
    return res;
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(std::initializer_list<unsigned int>&& dims, value_type init)
    : dim_(get_array(std::move(dims))),
      stride_(dim_),
      data_(new value_type[std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<unsigned int>())])
  {
    unsigned int count = std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<unsigned int>());
    for (unsigned int i = 0; i < count; i++)
      data_[i] = init;
  }


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(std::initializer_list<unsigned int>&& dims, Uninitialized<value_type>)
    : dim_(get_array(std::move(dims))),
      stride_(dim_),
      data_(new value_type[std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<unsigned int>())])
  { }

  /// ....
  explicit TensorSlowCpu(std::initializer_list<unsigned int>&& dims,
                         std::initializer_list<unsigned int>&& strides,
                         value_type init)
    : dim_(get_array(std::move(dims))),
      stride_(get_array(std::move(strides))),
      data_(new value_type[std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<unsigned int>())])
  {
    unsigned int count = std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<unsigned int>());
    for (unsigned int i = 0; i < count; i++)
      data_[i] = init;
  }

  explicit TensorSlowCpu(std::initializer_list<unsigned int>&& dims,
                         std::initializer_list<unsigned int>&& strides,
                         Uninitialized<value_type>)
    : dim_(get_array(std::move(dims))),
      stride_(get_array(std::move(strides))),
      data_(new value_type[std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<unsigned int>())])
  {
  }


  // Copy constructor
  // TODO: simple copy; implement reference counted buffers
  TensorSlowCpu(const TensorSlowCpu& other)
    : dim_{other.dim_},
      stride_{other.stride_},
      data_(new value_type[std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<unsigned int>())])
  {
    memcpy(data_,
           other.data_,
           std::accumulate(std::begin(stride_), std::end(stride_), sizeof(value_type), std::multiplies<unsigned int>()));
  }

  // Move constructor
  TensorSlowCpu(TensorSlowCpu&& other)
    : dim_{other.dim_},
      stride_{other.stride_},
      data_(std::move(other).data_)
  {
    other.data_ = nullptr;
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
  value_type* Data() const                        { return data_; }


  std::array<unsigned int, _Rank> dim_;
  std::array<unsigned int, _Rank> stride_;
  value_type* data_;
};


/// TensorSlowCpu<1, _T, _N> is a specialization of a rank-1 tensor (vector) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _N>
struct TensorSlowCpu<1, _T, _N> : TensorBase
{
  using value_type = _T;

  // helper function to initialize the std:array from an initializer list
  constexpr std::array<value_type, _N> get_array(std::initializer_list<value_type>&& init)
  {
    std::array<value_type,_N> res;
    std::copy(init.begin(), init.end(), res.begin());
    return res;
  }

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit TensorSlowCpu(std::initializer_list<value_type>&& init)
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
  const value_type* Data() const                  { return array_.data(); }


  unsigned int dim_[1];
  unsigned int stride_[1];
  std::array<value_type, _N> array_;
};


/// TensorSlowCpu<_T, _M, _N> is a specialization of a rank-2 tensor (matrix) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _M, size_t _N>
struct TensorSlowCpu<2, _T, _M, _N> : TensorBase
{
  using value_type = _T;

  // helper function to initialize the std:array from an initializer list
  constexpr std::array<value_type, _M * _N>
  get_array(std::initializer_list<std::initializer_list<value_type>>&& init)
  {
    std::array<value_type, _M * _N> res{};
    auto line_it = res.begin();
    for (auto it : init)
    {
      std::copy(it.begin(), it.end(), line_it);
      line_it += _N;
    }
    return res;
  }

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit TensorSlowCpu(std::initializer_list<std::initializer_list<value_type>>&& init)
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
  const value_type* Data() const                  { return array_.data(); }


  unsigned int dim_[2];
  unsigned int stride_[2];
  std::array<value_type, _M * _N> array_;
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

template <typename _Tensor> concept TensorSlowCpuType = is_same_runtime_v<_Tensor, TensorSlowCpu>;

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
