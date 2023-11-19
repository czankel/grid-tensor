//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_TENSOR_H
#define GRID_TENSOR_BASE_TENSOR_H

#include <array>
#include <initializer_list>

#include "../array.h"
#include "../tensor_parameters.h"

#include "copy.h"

namespace grid {

/// Tensor provides an non-optimized base implementation of tensors.
/// Note that the implementation implicitly requires that the buffer and strides are aligned to the value type.
template <typename _T, size_t _Rank, typename Allocator=std::allocator<_T>>
class Tensor
{
 public:
  using value_type = _T;
  using allocator_type = Allocator;
  using pointer = _T*;
  using const_pointer = const _T*;
  constexpr static size_t rank = _Rank;

 private:
  inline void
  initialize(pointer ptr,
             std::span<size_t, 1> dims,
             std::span<ssize_t, 1> strides,
             value_type init)
  {
    for (size_t i = 0; i < dims[0]; i++, ptr = ptr + strides[0]/sizeof(value_type))
      *ptr = init;
  }

  template <size_t _N>
  inline void
  initialize(pointer ptr,
             std::span<size_t, _N> dims,
             std::span<ssize_t, _N> strides,
             value_type init)
  {
    for (size_t i = 0; i < dims[0]; i++, ptr += strides[0]/sizeof(value_type))
      initialize(ptr,
                 std::span<size_t, _N - 1>(dims.begin() + 1, dims.end()),
                 std::span<ssize_t, _N - 1>(strides.begin() + 1, strides.end()),
                 init);
  }

 public:
  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  explicit Tensor(size_t dim, value_type init)
    : dims_{dim},
      strides_{make_strides<value_type>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dim, Uninitialized<value_type>)
    : dims_{dim},
      strides_{make_strides<value_type>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit Tensor(size_t dim_m, int dim_n, value_type init)
    : dims_{dim_m, dim_n},
      strides_{make_strides<value_type>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dim_m, int dim_n, Uninitialized<value_type>)
    : dims_{dim_m, dim_n},
      strides_{make_strides<value_type>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(std::initializer_list<size_t>&& dims, value_type init)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_{make_strides<value_type>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(std::initializer_list<size_t>&& dims, Uninitialized<value_type>)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_{make_strides<value_type>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with strides.
  explicit Tensor(std::initializer_list<size_t>&& dims,
                  std::initializer_list<ssize_t>&& strides,
                  value_type init)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_(get_array<ssize_t, _Rank>(std::move(strides))),
      size_(get_buffer_size(dims_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with strides.
  explicit Tensor(std::initializer_list<size_t>&& dims,
                  std::initializer_list<ssize_t>&& strides,
                  Uninitialized<value_type>)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_(get_array<ssize_t, _Rank>(std::move(strides))),
      size_(get_buffer_size(dims_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(const size_t(&dim)[_Rank], const ssize_t(&stride)[_Rank], value_type init)
    : dims_(get_array<size_t, _Rank>(dim)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      size_(get_buffer_size(dims_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  explicit Tensor(const size_t(&dim)[_Rank], const ssize_t(&stride)[_Rank], Uninitialized<value_type>)
    : dims_(get_array<size_t, _Rank>(dim)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      size_(get_buffer_size(dims_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer.
  explicit Tensor(std::array<size_t, _Rank> dims, value_type init)
    : dims_(dims),
      strides_(make_strides<value_type>(dims)),
      size_(get_buffer_size(dims_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize<_Rank>(data_, dims_, strides_, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with padding.
  explicit Tensor(std::array<size_t, _Rank> dims,
                  std::array<ssize_t, _Rank> strides,
                  value_type init)
    : dims_{dims},
      strides_{strides},
      size_(get_buffer_size(dims_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize<_Rank>(data_, dims_, strides_, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer.
  /// Note: assumes strides are type-aligned.
  explicit Tensor(std::array<size_t, _Rank> dims, Uninitialized<value_type>)
    : dims_{dims},
      strides_{make_strides<value_type>(dims)},
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with padding.
  explicit Tensor(std::array<size_t, _Rank> dims,
                  std::array<ssize_t, _Rank> strides,
                  Uninitialized<value_type>)
    : dims_{dims},
      strides_{strides},
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  Tensor(Tensor&& other)
    : dims_{std::move(other.dims_)},
      strides_{std::move(other.strides_)},
      size_(std::move(other.size_)),
      data_(std::move(other.data_))
  {
    other.data_ = nullptr;
  }

  Tensor(const Tensor& other)
    : dims_{other.Dimensions()},
      strides_{other.Strides()},
      size_(other.Size()),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    copy<value_type, _Rank>(data_, other.Data(), dims_, strides_, other.Strides());
  }

  // Constructors for converting from a tensor operator.
  template <AnyOperator Operator>
  Tensor(Operator&& functor) : Tensor{std::move(functor())} {};

  template <AnyOperator Operator>
  Tensor(const Operator& functor) : Tensor{functor()} {};


  /// Destructor
  ~Tensor()
  {
    if (data_ != nullptr)
      delete[] data_;
  }


  /// Assign operator
  template <PrimitiveTensor _Tensor>
  Tensor& operator=(const _Tensor& other)
  {
    dims_ = other.Dimensions();
    strides_ = make_strides<value_type>(dims_);
    size_ = strides_[0] * dims_[0];
    if (data_ != nullptr)
      delete[] data_;
    data_ = new value_type[size_ / sizeof(value_type)];
    copy<value_type, _Rank>(data_, other.Data(), dims_, strides_, other.Strides());
    return *this;
  }

  /// Move-assign is only supported from the same type
  Tensor& operator=(Tensor&& other)
  {
    dims_ = other.Dimensions();
    strides_ = make_strides<value_type>(dims_);
    size_ = strides_[0] * dims_[0];
    if (data_ != nullptr)
      delete[] data_;
    data_ = other.Data();
    return *this;
  }

  template <AnyOperator _Operator>
  Tensor& operator=(_Operator&& oper)
  {
    return operator=(std::forward<_Operator>(oper)());
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, _Rank>& Dimensions() const     { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return data_; }
  const_pointer Data() const                              { return data_; }

 private:
  std::array<size_t, _Rank>         dims_;
  std::array<ssize_t, _Rank>        strides_;
  size_t                            size_;
  pointer                           data_;
};

/// Tensor<_T, 0, 1> is a specialization of a rank-0 tensor.
template <typename _T>
class Tensor<_T, 0>
{
 public:
  using value_type = _T;
  using allocator_type = StaticAllocator<1>;
  using pointer = _T*;
  using const_pointer = const _T*;
  constexpr static size_t rank = 0UL;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit Tensor(value_type init) : data_{init} {}

  explicit Tensor(Uninitialized<value_type>) {}

  explicit Tensor(const std::array<size_t, 0>&, value_type init) : data_{init} {}

  explicit Tensor(const std::array<size_t, 0>&, Uninitialized<value_type>) {}

  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 0UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 0>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 0>& Strides() const           { return {strides_}; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(_T); }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return &data_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return &data_; }

 private:
  constexpr static std::array<size_t, 0>  dims_{};
  constexpr static std::array<ssize_t, 0> strides_ = {};
  value_type                              data_;
};


/// Tensor<_T, 1, _N> is a specialization of a rank-1 tensor (vector) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _N>
class Tensor<_T, 1, StaticAllocator<_N>>
{
 public:
  using value_type = _T;
  using allocator_type = StaticAllocator<_N>;
  using pointer = const _T*;
  using const_pointer = const _T*;
  constexpr static size_t rank = 1UL;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit Tensor(std::initializer_list<value_type>&& init)
    : dims_{_N},
      strides_{sizeof(value_type)},
      array_(get_array<value_type, _N>(std::move(init)))
  {}

  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 1UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 1>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 1>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type) * _N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }

 private:
  std::array<size_t, 1>             dims_;
  std::array<ssize_t, 1>            strides_;
  std::array<value_type, _N>        array_;
};


/// Tensor<_T, _M, _N> is a specialization of a rank-2 tensor (matrix) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _M, size_t _N>
class Tensor<_T, 2, StaticAllocator<_M, _N>>
{
 public:
  using value_type = _T;
  using pointer = const _T*;
  using const_pointer = const _T*;
  constexpr static size_t rank = 2UL;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit Tensor(std::initializer_list<std::initializer_list<value_type>>&& init)
    : dims_{_M, _N},
      strides_{ sizeof(value_type) * _N, sizeof(value_type)},
      array_(get_array<value_type, _M, _N>(std::move(init)))
  {}

  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 2UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 2>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 2>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type) * _M * _N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }

 private:
  std::array<size_t, 2>             dims_;
  std::array<ssize_t, 2>            strides_;
  std::array<value_type, _M * _N>   array_;
};


/// Tensor<_T, _Rank, NoAllocator> is a tensor for an externally managed buffer
template <typename _T, size_t _Rank>
class Tensor<_T, _Rank, NoAllocator>
{
 public:
  using value_type = _T;
  using allocator_type = NoAllocator;
  using pointer = const _T*;
  using const_pointer = const _T*;
  constexpr static size_t rank = _Rank;

  explicit Tensor() {}

  explicit Tensor(const ArrayView<value_type, _Rank>& arr)
    : dims_(arr.dims_),
      strides_(arr.Strides()),
      size_(arr.Size()),
      data_(arr.Data())
  {}

  explicit Tensor(ArrayView<value_type, _Rank>&& arr)
    : dims_(std::move(arr.Dimensions())),
      strides_(std::move(arr.Strides())),
      size_(arr.Size()),
      data_(std::move(arr.Data()))
  {}


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, _Rank>& Dimensions() const     { return dims_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size()                                           { return size_; }

  /// Data returns a pointer to the data buffer.
  char* Data() const                                      { return data_; }

 private:
  std::array<size_t, _Rank>   dims_;
  std::array<ssize_t, _Rank>  strides_;
  size_t                      size_;
  pointer                     data_;
};


// CTAD rules

// Tensor{T} -> Rank-0 tensor with a static/local array
template <typename _T>
explicit Tensor(_T) -> Tensor<_T, 0>;

// Tensor{Uninitailzied<T>} -> Rank-0 tensor with a static/local array
template <typename _T>
explicit Tensor(Uninitialized<_T>) -> Tensor<_T, 0>;

// Tensor{Ts...} -> Rank-1 tensor with a static/local array (brace-initializer).
template <typename _T, typename... _Ts>
explicit Tensor(_T, _Ts...) -> Tensor<std::common_type_t<_T, _Ts...>, 1, StaticAllocator<sizeof...(_Ts)+1>>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <typename _T, size_t... _N>
Tensor(_T(&&... l)[_N]) -> Tensor<_T, 2, StaticAllocator<sizeof...(_N), std::max({_N...})>>;


// Tensor(uint, _T) -> Rank-1 tensor with a dynamically allocated buffer.
template <typename _T>
explicit Tensor(size_t, _T) -> Tensor<_T, 1>;

// Tensor(uint, Uninitialized<T>) -> Rank-1 tensor with a dynamically allocated uninitialized buffer.
template <typename _T>
explicit Tensor(size_t, Uninitialized<_T>) -> Tensor<_T, 1>;

// Tensor(uint, uint, _T) -> Rank-2 tensor with a dynamically allocated buffer.
template <typename _T>
explicit Tensor(size_t, size_t, _T) -> Tensor<_T, 2>;

// Tensor(uint, Uninitialized<T>) -> Rank-2 tensor with a dynamically allocated uninitialized buffer.
template <typename _T>
explicit Tensor(size_t, size_t, Uninitialized<_T>) -> Tensor<_T, 2>;


// Tensor(&[], &[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _T, size_t _N>
explicit Tensor(const size_t(&)[_N], const ssize_t(&)[_N], _T) -> Tensor<_T, _N>;

// Tensor(&[], &[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _T, size_t _N>
explicit Tensor(const size_t(&)[_N], const ssize_t(&)[_N], Uninitialized<_T>) -> Tensor<_T, _N>;

// Tensor(&&[], &&[], _T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _T, size_t _N>
explicit Tensor(size_t(&&)[_N], ssize_t(&&)[_N], _T) -> Tensor<_T, _N>;

// Tensor(&&[], &&[]) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _T, size_t _N>
explicit Tensor(size_t(&&)[_N], ssize_t(&&)[_N], Uninitialized<_T>) -> Tensor<_T, _N>;


// Tensor(&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _T, size_t _N>
explicit Tensor(const size_t(&)[_N], _T) -> Tensor<_T, _N>;

// Tensor(&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _T, size_t _N>
explicit Tensor(const size_t(&)[_N], Uninitialized<_T>) -> Tensor<_T, _N>;

// Tensor(&&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _T, size_t _N>
explicit Tensor(const size_t(&&)[_N], _T) -> Tensor<_T, _N>;

// Tensor(&&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _T, size_t _N>
explicit Tensor(const size_t(&&)[_N], Uninitialized<_T>) -> Tensor<_T, _N>;


// Tensor(array, T)
template <typename _T, size_t _N>
Tensor(std::array<size_t, _N>, _T) -> Tensor<_T, _N>;

// Tensor(array, array, T)
template <typename _T, size_t _N>
explicit Tensor(std::array<size_t, _N>, std::array<ssize_t, _N>, _T) -> Tensor<_T, _N>;

// Tensor(array, Uninitialized<T>)
template <typename _T, size_t _N>
explicit Tensor(std::array<size_t, _N>, Uninitialized<_T>) -> Tensor<_T, _N>;

// Tensor(array, array, Uninitialized<T>)
template <typename _T, size_t _N>
explicit Tensor(std::array<size_t, _N>, std::array<ssize_t, _N>, Uninitialized<_T>) -> Tensor<_T, _N>;

// Tensor<ArrayView> -> Rank-N tensor for an externally managed buffer
template <typename _T, size_t _N>
explicit Tensor(const ArrayView<_T, _N>&) -> Tensor<_T, _N, NoAllocator>;
template <typename _T, size_t _N>
explicit Tensor(ArrayView<_T, _N>&&) -> Tensor<_T, _N, NoAllocator>;

// TensorOp -> Tensor (move)
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, typename...> typename _TensorRT, typename _T, size_t _Rank, typename... _Tensors>
Tensor(_TensorOp<_TensorRT, _T, _Rank, _Tensors...>&&) -> Tensor<_T, _Rank>;

// TensorOp -> Tensor (copy)
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, typename...> typename _TensorRT, typename _T, size_t _Rank, typename... _Tensors>
Tensor(const _TensorOp<_TensorRT,_T,  _Rank, _Tensors...>&) -> Tensor<_T, _Rank>;

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_TENSOR_H
