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

#include <algorithm>
#include <array>
#include <bitset>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <span>

#include <grid/tensor/mmap.h>

#include "copy.h"

namespace grid {
namespace base {

template <AnyTensor, size_t> class TensorView;

/// Tensor is an implementation of an "AI Tensor" for vector and matrix operations, and follows
/// less the mathematical or physical definition.
///
/// @tparam _Tp         Integral type
/// @tparam _Rank       Rank of the tensor with 0: scalar, 1: vector, 2: matrix, etc.
/// @tparam _Allocator  Stateless allocator; defaults to std::allocator
///
/// The Tensor defines these types:
///   value_type        _Tp
///   pointer      _Tp* or memory specific, which would require a 'map/unmap'
///
/// The Tensor must provide the following methods for returning the dimensions and strides:
///   size_t                     Rank()
///   std::array<size_t, Rank>   Dimensions()
///   std::array<ssize_t, Rank>  Strides()
///
/// The Tensor must also either have a direct pointer to the date:
///   pointer Data()    // FIXME: might be hidden? Maybe use Pointer
///
/// Or, alternative, provide means to map and unmap the memory (with a penality):
///   pointer Map()
///   void         UnMap()
///
/// A tensor view is also a tensor but doesn't own the data buffer. Instead, it points to
/// the data buffer of another tensor. Because of the object lifetime of the tensor, it has
/// some restrictions:
///
///  - A view can only be assigned to a tensor (or tensor view).
///  - If a tensor (or tensor view is assigned to the view, the data is always copied to the
///    'view' area, even if the other object is a rvalue reference.
///
/// Allocators
///
///  Tensors include a template parameter for the "allocator". The concept expands the allocator
///  definition of the std library and includes allocators for statically allocated buffers
///  (constants), device memory, and a "view" into the buffer of another tensor.
///
///  The tensor includes the allocate_type typedef, which can be probe with std::uses_allocator
///
///  The "view" has specific objecct ownership and lifetime dependencies that comes with
///  the following restrictions:
///
///   - assigning a view to a tensor (or other view) *always* copies the data, even if the view
///     is an rvalue reference.
///   - a view cannot be declared..
///   - an rvalue-reference tensor input returns a tensor instead of a view
///   - as an rvalue-reference, the view must be copied to the destination tensor
///
template <typename _Tp, size_t _Rank, typename Allocator=std::allocator<_Tp>>
class Tensor
{
 public:
  using value_type = _Tp;
  using allocator_type = Allocator;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = _Rank;

  // FIXME: add constexpr for fast-stride

 private:
  // helper functions to initialize the memory buffer with a specified value of the value_type.
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
  Tensor() = default;

  ~Tensor()
  {
    if (data_ != nullptr)
      delete[] data_;
  }

  Tensor(Tensor&& other)
    : dims_{other.Dimensions()},
      strides_{other.Strides()},
      size_(other.Size()),
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
    details::copy<value_type, _Rank>(data_, other.Data(), dims_, strides_, other.Strides());
  }


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
      size_(dims_[0] * strides_[0]),
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
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(const size_t(&dim)[_Rank], const ssize_t(&stride)[_Rank], value_type init)
    : dims_(get_array<size_t, _Rank>(dim)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  explicit Tensor(const size_t(&dim)[_Rank], const ssize_t(&stride)[_Rank], Uninitialized<value_type>)
    : dims_(get_array<size_t, _Rank>(dim)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer.
  explicit Tensor(std::array<size_t, _Rank> dims, value_type init)
    : dims_(dims),
      strides_(make_strides<value_type>(dims)),
      size_(dims_[0] * strides_[0]),
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
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize<_Rank>(data_, dims_, strides_, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer.
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


  /// Constructor from a 'trivially copyable' tensor.
  template <PrimitiveTensor _Tensor>
  Tensor(const _Tensor& other)
    : dims_(other.Dimensions()),
      strides_(other.Strides()),
      size_(dims_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    details::copy<value_type, _Rank>(data_, other.Data(), dims_, strides_, other.Strides());
  }


  /// Constructors for converting a tensor operator.
  template <AnyOperator _Operator>
  Tensor(_Operator&& functor) : Tensor{std::move(functor())} {}

  /// Constructors for converting a tensor operator.
  template <AnyOperator _Operator>
  Tensor(const _Operator& functor) : Tensor{functor()} {}


  /// Assign operator
  //template <PrimitiveTensor _Tensor>
  template <typename _Tensor>
  Tensor& operator=(const _Tensor& other)
  {
    dims_ = other.Dimensions();
    strides_ = make_strides<value_type>(dims_);
    size_ = strides_[0] * dims_[0];
    if (data_ != nullptr)
      delete[] data_;
    data_ = new value_type[size_ / sizeof(value_type)];
    details::copy<value_type, _Rank>(data_, other.Data(), dims_, strides_, other.Strides());
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

  /// Operator  FIXME: PrimitiveTensor  template <AnyOperator _Operator>
  template <AnyOperator _Operator>
  Tensor& operator=(_Operator&& oper)
  {
    return operator=(std::forward<_Operator>(oper)());
  }

  // FIXME: provide const and non-const versions

  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank] = {0})/* FIXME cannot be  const?qboth   */ &
  {
    return TensorView(*this, axes, offsets);
  }

  // FIXME: why not a const version? Tensor = const View?
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank]) const && = delete;


  /// Assign is an assignment to a view of this tensor.
  template <size_t _ViewRank, PrimitiveTensor _Tensor> // FIXME requires _ViewRank == Tensor.Rank()
  auto Assign(size_t offset,
              const std::array<size_t, _ViewRank>& dims, //const std::span<const size_t,  _ViewRank> dims,
              const std::array<ssize_t, _ViewRank>& strides, //const std::span<const ssize_t, _ViewRank> strides,
              const _Tensor& source)
  {
    // FIXME: dest.Dimensions() == source.Dimensions()
    // FIXME: dims and strides should already match, check here or in copy??
    details::copy<value_type, _ViewRank>(data_ + offset, source.Data(), dims, strides, source.Strides());
  }
  // FIXME add Assign for Tensor&? or std::forward?


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, _Rank>& Dimensions() const     { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return data_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return data_; }


  std::array<size_t, _Rank>         dims_;
  std::array<ssize_t, _Rank>        strides_;
  size_t                            size_;
  pointer                      data_;
};


/// Tensor<_Tp, 0, 1> is a specialization of a rank-0 tensor (scalar).
template <typename _Tp>
class Tensor<_Tp, 0>   // FIXME: StaticAllocator<1>?
{
 public:
  using value_type = _Tp;
  using allocator_type = StaticAllocator<0>;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = 0UL;;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit Tensor(_Tp init) : array_{init} {}

  explicit Tensor(Uninitialized<_Tp>) {}

  explicit Tensor(const std::array<size_t, 0>&, _Tp init) : array_{init} {}

  explicit Tensor(const std::array<size_t, 0>&, Uninitialized<_Tp>) {}


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 0UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 0>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 0>& Strides() const           { return {strides_}; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(_Tp); }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                            { return array_.data(); }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                                { return array_.data(); }

  constexpr static std::array<size_t, 0>  dims_{};
  constexpr static std::array<ssize_t, 0> strides_ = {};
  std::array<_Tp, 1>               array_;  // FIXME: why array?
};


/// Tensor<_Tp, 1, _N> is a specialization of a rank-1 tensor (vector) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _Tp, size_t _N>
class Tensor<_Tp, 1, StaticAllocator<_N>>
{
 public:
  using value_type = _Tp;
  using allocator_type = StaticAllocator<_N>;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = 1UL;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit Tensor(std::initializer_list<_Tp>&& init)
    : dims_{_N},
      strides_{sizeof(_Tp)},
      array_(get_array<_Tp, _N>(std::move(init)))
  {}

  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[1]) const &
  {
    return TensorView(*this, axes, offsets);
  }

  // FIXME: provide note or something, why this is prohibited..
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[1]) && = delete;


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 1UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 1>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 1>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(_Tp) * _N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }


  std::array<size_t, 1>       dims_;
  std::array<ssize_t, 1>      strides_;
  std::array<_Tp, _N>  array_;
};


/// Tensor<_Tp, _M, _N> is a specialization of a rank-2 tensor (matrix) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _Tp, size_t _M, size_t _N>
class Tensor<_Tp, 2, StaticAllocator<_M, _N>>
{
 public:
  using value_type = _Tp;
  using allocator_type = StaticAllocator<_M, _N>;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = 2UL;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit Tensor(std::initializer_list<std::initializer_list<_Tp>>&& init)
    : dims_{_M, _N},
      strides_{ sizeof(_Tp) * _N, sizeof(_Tp)},
      array_(get_array<_Tp, _M, _N>(std::move(init)))
  {}

  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[2]) const &
  {
    return TensorView(*this, axes, offsets);
  }

  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[2]) && = delete;


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 2UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 2>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 2>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(_Tp) * _M * _N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }


  std::array<size_t, 2>     dims_;
  std::array<ssize_t, 2>    strides_;
  std::array<_Tp, _M * _N>  array_;
};


/// Tensor<_Tp, _C, _M, _N> is a specialization of a rank-3 tensor for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _Tp, size_t _C, size_t _M, size_t _N>
class Tensor<_Tp, 3, StaticAllocator<_C, _M, _N>>
{
 public:
  using value_type = _Tp;
  using allocator_type = StaticAllocator<_C, _M, _N>;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = 3UL;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit Tensor(std::initializer_list<std::initializer_list<std::initializer_list<_Tp>>>&& init)
    : dims_{_C, _M, _N},
      strides_{ sizeof(_Tp) * _M * _N, sizeof(_Tp) * _N, sizeof(_Tp)},
      array_(get_array<_Tp, _C, _M, _N>(std::move(init)))
  {}

  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[3]) const &
  {
    return TensorView(*this, axes, offsets);
  }

  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[3]) && = delete;


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 3UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 3>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 3>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(_Tp) * _C * _M * _N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }


  std::array<size_t, 3>         dims_;
  std::array<ssize_t, 3>        strides_;
  std::array<_Tp, _C * _M * _N> array_;
};


/// Tensor<_Tp, _Rank, MemoryMapped{}> is a tensor for memory-mapped data
template <typename _Tp, size_t _Rank>
class Tensor<_Tp, _Rank, MemoryMapped<_Tp>>
{
 public:
  using value_type = _Tp;
  using allocator_type = MemoryMapped<_Tp>;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = _Rank;

  explicit Tensor() {}

  /// Constructor for a memory-mapped buffer.
  explicit Tensor(const MMapArray<_Tp, _Rank>& arr)
    : //size_(strides_[0] * dims_[0]),
      dims_(arr.dims_),
      strides_(arr.strides_),
      mmap_(arr.mmap_),
      data_(static_cast<pointer>(mmap_->Address()) + arr.offset_)
  {}

  // Constructor for a memory-mapped buffer.
  explicit Tensor(MMapArray<_Tp, _Rank>&& arr)
    : dims_(std::move(arr.dims_)),
      strides_(std::move(arr.strides_)),
      mmap_(std::move(arr.mmap_)),
      data_(static_cast<pointer>(mmap_->Address()) + arr.offset_)
  {}

  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank]) const &
  {
    return TensorView(*this, axes, offsets);
  }

  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank]) && = delete;


#if 0
  // FIXME: support operator= on memorymapped? need two versions, RO and RW?
  Tensor& operator=(Tensor&& other)
  {
    dims_ = std::move(other.dims_);
    strides_ = std::move(other.strides_);
    mmap_ = std::move(other.mmap_);
    data_ = other.data_;
    return *this;
  }

  Tensor& operator=(const Tensor& other)
  {
    dims_ = other.dims_;
    strides_ = other.strides_;
    mmap_ = other.mmap_;
    data_ = other.data_;
    return *this;
  }
#endif


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, _Rank>& Dimensions() const     { return dims_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size() const                                     { return mmap_->Size(); }

  /// Data returns a pointer to the data buffer. // FIXME: need const version?
  pointer Data()                                            { return data_; }

  const_pointer Data() const                                { return data_; }


  size_t                      size_;
  std::array<size_t, _Rank>   dims_;
  std::array<ssize_t, _Rank>  strides_;
  std::shared_ptr<MMap>       mmap_;
  pointer                       data_;
};

#if 0
/// ViewAllocator is a special "allocator" for  for referencing a region of a referenced tensor buffer.
/// It has specific ownership and lifetime dependencies. More information under tensor.
struct ViewAllocator {};
#endif


/// TensorView<Tensor, _Rank> is a view of a tensor.
///
/// A view cannot be created from a temporary rval; it will return a tensor
/// FIXME:
///   View&& view;    << requires initializer
///   View& view;     << requires initializer
///
template <AnyTensor _Tensor, size_t _Rank>
class TensorView
{
 public:
  // FIXME
  using value_type = typename _Tensor::value_type;
  using pointer = typename _Tensor::pointer;
  using const_pointer = typename _Tensor::const_pointer;
  constexpr static size_t rank = _Rank;

  /// TensorView must be in the same scope and lifetime as the underlying tensor.
  TensorView() = delete;

#if 0 // FIXME: what's this?
  template <typename Tensor, size_t TensorRank>
  explicit TensorView(const Tensor& tensor, const ssize_t(& axes)[_Rank])
    : Tensor(tensor, axes, { 0 })
  {}
#endif

  // using Tensor = CUDA::Tensor; --> what about TensorView?? Or ... TensorConst TensorMMap?
  // using Tensor = TensorCUDA;
  // using Tensor = Tensor !?

  // FIXME: this is all common? --> grid::TensorView
  // except for Data()??? use Tensor::pointer and keep offset?
  // would an implementation need to check alignment and override this View?
  // In the end, View's cannot really be defined, only thourhg a Tensor, so could be overridden?
  // Unless:   grid::View(tensor, {0, 1}); instead of tensor.View({0,1}
  // What about "Viewable"?
  // Can also check if Tensor::pointer is an integral pointer? 
// FIXME: why have to do this over and over again? should be able to do something like this?
//    tensor::View input_view(input_tensor, dims, ..., offset); // FIXME: tensor must stay alive!!
//
//    View view;
//    {
//      input_tensor;
//      view = View(intput_tensor) !!! error cannot be assigned...
//    }
//
  template </*typename Tensor, */ size_t TensorRank>
  explicit TensorView(_Tensor& tensor, const ssize_t(& axes)[_Rank], const ssize_t(& offsets)[TensorRank])
    : tensor_(tensor)
 {
    std::bitset<_Rank> handled = false;
    auto strides = tensor.Strides();
    auto dims    = tensor.Dimensions();

    // FIXME: strides is wrong:   4,1 (1x, 1x) -> 1,4,1 (4x,1x,1x)!! the multiplication is not encoded..
    for (ssize_t i = static_cast<ssize_t>(_Rank) - 1; i >= 0; i--)
    {
      // FIXME: dims[axes[i]] could already be one? doesn't really matter...
      if (axes[i] >= 0)
      {
        if (handled[axes[i]])
          throw std::runtime_error("axis can only be used once");

        handled[axes[i]] = true;// FIXNE: handled.set(...);
        dims_[i] = dims[axes[i]];
        strides_[i] = strides[axes[i]];
      }
      else if (axes[i] == Broadcast)
      {
        dims_[i] = 1;
        strides_[i] = 0; // i < static_cast<ssize_t>(_Rank)-1 ? strides_[i+1] : sizeof(_Tp);
        //if (i == 0)
         // strides_[0] = tensor.Size();
      }
      else
        throw std::runtime_error("Invalid axis");
    }


    size_t offset = 0;
    for (size_t i = 0; i < TensorRank; i++)
      offset += offsets[i] * tensor.strides_[i];
    // FIXME: check that offset is less than dims?

    data_ = const_cast<pointer>(tensor.Data()) + offset;
    offset_ = offset;
  }

  template <AnyTensor _FromTensor> requires (_FromTensor::rank == _Rank) // TensorRank<_FromTensor, _Rank> // FIXME : can this be combined?? Does it need to be viewable? requires Viewable?
  auto operator=(const _FromTensor& source)
  {
    tensor_.Assign(offset_, dims_, strides_, source);
    return *this;
  }




  // FIXME: pass-through to Tensor? Makes View more  ... somehow?
#if 0 // FIXME implement tensor operato= or pass-throught o tensor, ... Tensor::CopyView=(this, other)??
  // TODO: views, check boundaries
  template <Viewable _View> requires TensorRank<_View, _Rank> // FIXME : can this be combined?? Does it need to be viewable? requires Viewable?
  auto operator=(const _View& view)
  {
    details::copy<_Tp, _Rank>(data_, view.Data(), dims_, strides_, view.Strides());
    return *this;
  }

  template <PrimitiveTensor _View> requires TensorRank<_View, _Rank>
  auto operator=(_View&& view)
  {
    details::copy<_Tp, _Rank>(data_, view.Data(), dims_, strides_, view.Strides());
    return *this;
  }

  template <PrimitiveTensor _View _View> requires TensorRank<_View, _Rank>
  auto operator=(const _View& view) const = delete;

  template <PrimitiveTensor _View _View> requires TensorRank<_View, _Rank>
  auto operator=(_View&&) const = delete;
#endif

  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, _Rank>& Dimensions() const     { return dims_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size()                                           { return strides_[0] * dims_[0]; } // FIXME: wrong

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return data_; }

  const_pointer Data() const                              { return data_; }

  _Tensor&                    tensor_;

  std::array<size_t, _Rank>   dims_;
  std::array<ssize_t, _Rank>  strides_;

  // FIXME: should it copy data field_
  pointer                     data_;
  size_t                      offset_;
};

#if 0
// FIXME: need to 'amend' with rank-2 0s...
  auto Row(size_t index)                                  { return View({1}, {index, ....}); }
  auto Col(size_t index)                                  { return View({0}, {0, index, 0}); }
#endif


// CTAD rules

// Tensor{T} -> Rank-0 tensor with a static/local array
template <typename _Tp>
explicit Tensor(_Tp) -> Tensor<_Tp, 0>;

// Tensor{Uninitailzied<_Tp>} -> Rank-0 tensor with a static/local array
template <typename _Tp>
explicit Tensor(Uninitialized<_Tp>) -> Tensor<_Tp, 0>;

// Tensor{_Ts...} -> Rank-1 tensor with a static/local array (brace-initializer).
template <typename _Tp, typename... _Ts>
explicit Tensor(_Tp, _Ts...) -> Tensor<std::common_type_t<_Tp, _Ts...>, 1, StaticAllocator<sizeof...(_Ts)+1>>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <typename _Tp, size_t... _N>
Tensor(_Tp(&&... l)[_N]) -> Tensor<_Tp, 2, StaticAllocator<sizeof...(_N), std::max({_N...})>>;

// Tensor{{{...},...},...} -> Rank-3 tensor with a static/local array (brace-initializer).
template <typename _Tp, size_t... _M, size_t... _N>
explicit Tensor(_Tp(&&... l)[_M][_N]) -> Tensor<_Tp, 3, StaticAllocator<sizeof...(_M), std::max({_M...}), std::max({_N...})>>;


// Tensor(uint, T) -> Rank-1 tensor with a dynamically allocated buffer.
template <typename _Tp>
explicit Tensor(size_t, _Tp) -> Tensor<_Tp, 1>;

// Tensor(uint, Uninitialized<T>) -> Rank-1 tensor with a dynamically allocated uninitialized buffer.
template <typename _Tp>
explicit Tensor(size_t, Uninitialized<_Tp>) -> Tensor<_Tp, 1>;

// Tensor(uint, uint, T) -> Rank-2 tensor with a dynamically allocated buffer.
template <typename _Tp>
explicit Tensor(size_t, size_t, _Tp) -> Tensor<_Tp, 2>;

// Tensor(uint, Uninitialized<T>) -> Rank-2 tensor with a dynamically allocated uninitialized buffer.
template <typename _Tp>
explicit Tensor(size_t, size_t, Uninitialized<_Tp>) -> Tensor<_Tp, 2>;


// Tensor(&[], &[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _Tp, size_t _N>
explicit Tensor(const size_t(&)[_N], const ssize_t(&)[_N], _Tp) -> Tensor<_Tp, _N>;

// Tensor(&[], &[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _Tp, size_t _N>
explicit Tensor(const size_t(&)[_N], const ssize_t(&)[_N], Uninitialized<_Tp>) -> Tensor<_Tp, _N>;

// Tensor(&&[], &&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _Tp, size_t _N>
explicit Tensor(size_t(&&)[_N], ssize_t(&&)[_N], _Tp) -> Tensor<_Tp, _N>;

// Tensor(&&[], &&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _Tp, size_t _N>
explicit Tensor(size_t(&&)[_N], ssize_t(&&)[_N], Uninitialized<_Tp>) -> Tensor<_Tp, _N>;


// Tensor(&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _Tp, size_t _N>
explicit Tensor(const size_t(&)[_N], _Tp) -> Tensor<_Tp, _N>;

// Tensor(&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _Tp, size_t _N>
explicit Tensor(const size_t(&)[_N], Uninitialized<_Tp>) -> Tensor<_Tp, _N>;

// Tensor(&&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _Tp, size_t _N>
explicit Tensor(const size_t(&&)[_N], _Tp) -> Tensor<_Tp, _N>;

// Tensor(&&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _Tp, size_t _N>
explicit Tensor(const size_t(&&)[_N], Uninitialized<_Tp>) -> Tensor<_Tp, _N>;


// Tensor(array, T)
template <typename _Tp, size_t _N>
Tensor(std::array<size_t, _N>, _Tp) -> Tensor<_Tp, _N>;

// Tensor(array, array, T)
template <typename _Tp, size_t _N>
explicit Tensor(std::array<size_t, _N>, std::array<ssize_t, _N>, _Tp) -> Tensor<_Tp, _N>;

// Tensor(array, Uninitialized<T>)
template <typename _Tp, size_t _N>
explicit Tensor(std::array<size_t, _N>, Uninitialized<_Tp>) -> Tensor<_Tp, _N>;

// Tensor(array, array, Uninitialized<T>)
template <typename _Tp, size_t _N>
explicit Tensor(std::array<size_t, _N>, std::array<ssize_t, _N>, Uninitialized<_Tp>) -> Tensor<_Tp, _N>;


// Tensor<mmap, dim, strides> -> Rank-N tensor for a memory mapped buffer
template <typename _Tp, size_t _N>
explicit Tensor(MMapArray<_Tp, _N>) -> Tensor<_Tp, _N, MemoryMapped<_Tp>>;

#if 0
// Tensor(Tensor...) FIXME: needed? will it work??
template <typename _Tp, size_t _N>
explicit Tensor(Tensor<_Tp, 1, StaticAllocator<_N>>&) -> Tensor<_Tp, _N>;

template <typename _Tp, size_t _N, size_t _M>
explicit Tensor(Tensor<_Tp, 2, StaticAllocator<_N, _M>>&) -> Tensor<_Tp, _N * _M>;

template <typename _Tp, size_t _C, size_t _N, size_t _M>
explicit Tensor(Tensor<_Tp, 3, StaticAllocator<_C, _N, _M>>&) -> Tensor<_Tp, _C * _N * _M>;
#endif

// Tensor(Operator) -> Tensor (move)
template <template <template <typename, size_t, typename> typename, typename, size_t, typename...> typename _Operator,
          template <typename, size_t, typename> typename _Tensor, typename _Tp, size_t _Rank, typename... _Tensors>
Tensor(_Operator<_Tensor, _Tp, _Rank, _Tensors...>&&) -> Tensor<_Tp, _Rank>;

// Tensor(Operator) -> Tensor (copy)
template <template <template <typename, size_t, typename> typename, typename, size_t, typename...> typename _Operator,
          template <typename, size_t, typename> typename _Tensor, typename _Tp, size_t _Rank, typename... _Tensors>
Tensor(const _Operator<_Tensor, _Tp,  _Rank, _Tensors...>&) -> Tensor<_Tp, _Rank>;

} // end of namespace base
} // end of namespace grid

#endif  // GRID_TENSOR_BASE_TENSOR_H
