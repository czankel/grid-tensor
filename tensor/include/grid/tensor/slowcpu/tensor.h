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
#include <cstring>
#include <initializer_list>
#include <numeric>

namespace grid {

/// TensorSlowCpu<_T, _Rank> is a specialization of TensorSlowCpu for a dynamically allocated buffer.
/// Note that this is also the Tensor used for any TensorOp result.
template <typename _T, size_t _Rank>
struct TensorSlowCpu<_T, _Rank> : TensorBase
{
  using tensor_type = TensorSlowCpu<_T, _Rank>;
  using value_type = _T;

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  // TODO: initializes entire data buffer based on stride dimensions
  explicit TensorSlowCpu(size_t dim, value_type init)
    : dim_{dim}, stride_{dim}, data_(new value_type[dim])
  {
    size_t count = std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<size_t>());
    for (size_t i = 0; i < count; i++)
      data_[i] = init;
  }

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(size_t dim, Uninitialized<value_type>)
    : dim_{dim}, stride_{dim}, data_(new value_type[dim])
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit TensorSlowCpu(size_t dim_m, int dim_n, value_type init)
    : dim_{dim_m, dim_n},
      stride_{dim_m, dim_n},
      data_(new value_type[dim_m * dim_n])
  {
    size_t count = std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<size_t>());
    for (size_t i = 0; i < count; i++)
      data_[i] = init;
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(size_t dim_m, int dim_n, Uninitialized<value_type>)
  : dim_{dim_m, dim_n},
    stride_{dim_m, dim_n},
    data_(new value_type[dim_m * dim_n])
  {}

  // helper function to extra brace-initializer list
  constexpr std::array<size_t, _Rank> get_array(std::initializer_list<size_t>&& dim)
  {
    std::array<size_t, _Rank> res;
    std::copy(dim.begin(), dim.end(), res.begin());
    return res;
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims, value_type init)
    : dim_(get_array(std::move(dims))),
      stride_(dim_),
      data_(new value_type[std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<size_t>())])
  {
    size_t count = std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<size_t>());
    for (size_t i = 0; i < count; i++)
      data_[i] = init;
  }


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims, Uninitialized<value_type>)
    : dim_(get_array(std::move(dims))),
      stride_(dim_),
      data_(new value_type[std::accumulate(std::begin(dim_), std::end(dim_), 1, std::multiplies<size_t>())])
  { }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with strides.
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims,
                         std::initializer_list<size_t>&& strides,
                         value_type init)
    : dim_(get_array(std::move(dims))),
      stride_(get_array(std::move(strides))),
      data_(new value_type[std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<size_t>())])
  {
    size_t count = std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<size_t>());
    for (size_t i = 0; i < count; i++)
      data_[i] = init;
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with strides.
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims,
                         std::initializer_list<size_t>&& strides,
                         Uninitialized<value_type>)
    : dim_(get_array(std::move(dims))),
      stride_(get_array(std::move(strides))),
      data_(new value_type[std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<size_t>())])
  {}


  // Copy constructor
  // TODO: simple copy; implement reference counted buffers
  TensorSlowCpu(const TensorSlowCpu& other)
    : dim_{other.dim_},
      stride_{other.stride_},
      data_(new value_type[std::accumulate(std::begin(stride_), std::end(stride_), 1, std::multiplies<size_t>())])
  {
    memcpy(data_,
           other.data_,
           std::accumulate(std::begin(stride_), std::end(stride_), sizeof(value_type), std::multiplies<size_t>()));
  }

  // Move constructor
  TensorSlowCpu(TensorSlowCpu&& other)
    : dim_{other.dim_},
      stride_{other.stride_},
      data_(std::move(other).data_)
  {
    other.data_ = nullptr;
  }

  // Constructors for converting from a tensor operator.
  template <TensorOpFor<TensorSlowCpu> Operator> TensorSlowCpu(Operator&& op) : TensorSlowCpu{std::move(op())} {}
  template <TensorOpFor<TensorSlowCpu> Operator> TensorSlowCpu(const Operator& op) : TensorSlowCpu{op()} {}

  /// Destructor
  ~TensorSlowCpu()                                { delete[] data_; }

  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                  { return _Rank; }

  /// Dim returns the dimension of the rank.
  size_t Dim(size_t index) const                  { return dim_[index]; }

  // TODOo: assert on the index; also Dim()
  /// Stride returns the stride of the rank.
  size_t Stride(size_t index) const               { return stride_[index]; }

  /// Data returns a pointer to the data buffer.
  value_type* Data() const                        { return data_; }


  std::array<size_t, _Rank> dim_;
  std::array<size_t, _Rank> stride_;
  value_type*               data_;
};


/// TensorSlowCpu<_T, 1, _N> is a specialization of a rank-1 tensor (vector) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _N>
struct TensorSlowCpu<_T, 1, _N> : TensorBase
{
  using tensor_type = TensorSlowCpu<_T, 1, _N>;
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
  size_t Dim(size_t index) const                  { if (index > 1) throw std::out_of_range ("index");
                                                    return dim_[index]; }
  /// Dim returns the stride for the rank.
  size_t Stride(size_t index) const               { if (index > 1) throw std::out_of_range ("index");
                                                    return stride_[index]; }
  /// Data returns a pointer to the data buffer.
  const value_type* Data() const                  { return array_.data(); }


  size_t                      dim_[1];
  size_t                      stride_[1];
  std::array<value_type, _N>  array_;
};


/// TensorSlowCpu<_T, _M, _N> is a specialization of a rank-2 tensor (matrix) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _M, size_t _N>
struct TensorSlowCpu<_T, 2, _M, _N> : TensorBase
{
  using tensor_type = TensorSlowCpu<_T, 2, _M, _N>;
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
  size_t Dim(size_t index) const                  { if (index > 2) throw std::out_of_range ("index");
                                                    return dim_[index]; }
  /// Dim returns the stride of the rank.
  size_t Stride(size_t index) const               { if (index > 2) throw std::out_of_range ("index");
                                                    return stride_[index]; }
  /// Data returns a pointer to the data buffer.
  const value_type* Data() const                  { return array_.data(); }


  size_t                          dim_[2];
  size_t                          stride_[2];
  std::array<value_type, _M * _N> array_;
};


// CTAD rules

// Tensor{Ts...} -> Rank-1 tensor with a static/local array (brace-initializer).
template <typename _T, typename... _Ts>
explicit TensorSlowCpu(_T, _Ts...) -> TensorSlowCpu<std::common_type_t<_T, _Ts...>, 1, sizeof...(_Ts)+1>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <typename _T, size_t... _N>
TensorSlowCpu(_T(&&... l)[_N]) -> TensorSlowCpu<_T, 2, sizeof...(_N), std::max({_N...})>;


// Tensor(uint, _T) -> Rank-1 tensor with a dynamically allocated buffer.
template <typename _T>
explicit TensorSlowCpu(size_t, _T) -> TensorSlowCpu<_T, 1>;

// Tensor(uint, Uninitialized<T>) -> Rank-1 tensor with a dynamically allocated uninitialized buffer.
template <typename _T>
explicit TensorSlowCpu(size_t, Uninitialized<_T>) -> TensorSlowCpu<_T, 1>;

// Tensor(uint, uint, _T) -> Rank-2 tensor with a dynamically allocated buffer.
template <typename _T>
explicit TensorSlowCpu(size_t, size_t, _T) -> TensorSlowCpu<_T, 2>;

// Tensor(uint, Uninitialized<T>) -> Rank-2 tensor with a dynamically allocated uninitialized buffer.
template <typename _T>
explicit TensorSlowCpu(size_t, size_t, Uninitialized<_T>) -> TensorSlowCpu<_T, 2>;

// Tensor(uint[], uint[]) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _T, size_t _N>
explicit TensorSlowCpu(size_t(&&d)[_N], size_t(&&s)[_N], _T) -> TensorSlowCpu<_T, _N>;

// Tensor(uint[], uinit[]) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _T, size_t _N>
explicit TensorSlowCpu(size_t(&&d)[_N], size_t(&&s)[_N], Uninitialized<_T>) -> TensorSlowCpu<_T, _N>;

// TensorOp -> Tensor (move)
template <template <template <typename, size_t, auto...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, auto...> typename _TensorRT, typename _T, size_t _Rank, typename... _Tensors>
TensorSlowCpu(_TensorOp<_TensorRT, _T, _Rank, _Tensors...>&&) -> TensorSlowCpu<_T, _Rank>;

// TensorOp -> Tensor (copy)
template <template <template <typename, size_t, auto...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, auto...> typename _TensorRT, typename _T, size_t _Rank, typename... _Tensors>
TensorSlowCpu(const _TensorOp<_TensorRT,_T,  _Rank, _Tensors...>&) -> TensorSlowCpu<_T, _Rank>;


} // end of namespace grid

#endif  // GRID_TENSOR_SLOWCPU_TENSOR_H
