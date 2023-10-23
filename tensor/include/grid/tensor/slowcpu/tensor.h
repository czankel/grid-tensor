//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_SLOWCPU_TENSOR_H
#define GRID_TENSOR_SLOWCPU_TENSOR_H

#include <algorithm>
#include <array>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <span>

#include <grid/tensor/mmap.h>

#include "copy.h"

namespace grid {

/// TensorSlowCpu is an unoptimized tensor implementatoin for the CPU.
template <typename, size_t, auto...> struct TensorSlowCpu;

/// TensorSlowCpu<_T, _Rank> is a specialization of TensorSlowCpu for a dynamically allocated buffer.
///
/// Note that this is also the Tensor used for any TensorOp result.
/// TODO: see if constructors can be combined using implicit conversion
template <typename _T, size_t _Rank>
struct TensorSlowCpu<_T, _Rank>
{
  using value_type = std::decay_t<_T>;

  constexpr static size_t rank = _Rank;

  inline void
  initialize(char* ptr,
             std::span<size_t, 1> dims,
             std::span<ssize_t, 1> strides,
             value_type init)
  {
    for (size_t i = 0; i < dims[0]; i++, ptr += strides[0])
      *reinterpret_cast<value_type*>(ptr) = init;
  }

  template <size_t _N>
  inline void
  initialize(char* ptr,
             std::span<size_t, _N> dims,
             std::span<ssize_t, _N> strides,
             value_type init)
  {
    for (size_t i = 0; i < dims[0]; i++, ptr += strides[0])
      initialize(ptr,
                 std::span<size_t, _N - 1>(dims.begin() + 1, dims.end()),
                 std::span<ssize_t, _N - 1>(strides.begin() + 1, strides.end()),
                 init);
  }

  /// Default Constructor
  TensorSlowCpu() {}


  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  explicit TensorSlowCpu(size_t dim, value_type init)
    : dims_{dim},
      strides_{make_strides<_T>(dims_)},
      shared_(new char[dims_[0] * strides_[0]]),
      data_(shared_.get())
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(size_t dim, Uninitialized<value_type>)
    : dims_{dim},
      strides_{make_strides<_T>(dims_)},
      shared_(new char[dims_[0] * strides_[0]]),
      data_(shared_.get())
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit TensorSlowCpu(size_t dim_m, int dim_n, value_type init)
    : dims_{dim_m, dim_n},
      strides_{make_strides<_T>(dims_)},
      shared_(new char[dims_[0] * strides_[0]]),
      data_(shared_.get())
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(size_t dim_m, int dim_n, Uninitialized<value_type>)
    : dims_{dim_m, dim_n},
      strides_{make_strides<_T>(dims_)},
      shared_(new char[dims_[0] * strides_[0]]),
      data_(shared_.get())
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims, value_type init)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_{make_strides<_T>(dims_)},
      shared_(new char[dims_[0] * strides_[0]]),
      data_(shared_.get())
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims, Uninitialized<value_type>)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_{make_strides<_T>(dims_)},
      shared_(new char[dims_[0] * strides_[0]]),
      data_(shared_.get())
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with strides.
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims,
                         std::initializer_list<ssize_t>&& strides,
                         value_type init)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_(get_array<ssize_t, _Rank>(std::move(strides))),
      shared_(new char[dims_[0] * strides_[0]]),
      data_(shared_.get())
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with strides.
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims,
                         std::initializer_list<ssize_t>&& strides,
                         Uninitialized<value_type>)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_(get_array<ssize_t, _Rank>(std::move(strides))),
      shared_(new char[dims_[0] * strides_[0]]),
      data_(shared_.get())
  {}

  // FIXME: can this be combined with initializer-list?
  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(const size_t(&dim)[_Rank], const ssize_t(&stride)[_Rank], value_type init)
    : dims_(get_array<size_t, _Rank>(dim)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      shared_(new char[dims_[0] * strides_[0]]),
      data_(shared_.get())
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  explicit TensorSlowCpu(const size_t(&dim)[_Rank], const ssize_t(&stride)[_Rank], Uninitialized<_T>)
    : dims_(get_array<size_t, _Rank>(dim)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      shared_(new char[dims_[0] * strides_[0]]),
      data_(shared_.get())
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer.
  explicit TensorSlowCpu(std::array<size_t, _Rank> dims,
                         value_type init)
    : dims_(dims),
      strides_(make_strides<_T>(dims)),
      shared_(new char[strides_[0] * dims_[0]]),
      data_(shared_.get())
  {
    initialize<_Rank>(data_, dims_, strides_, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with padding.
  explicit TensorSlowCpu(std::array<size_t, _Rank> dims,
                         std::array<ssize_t, _Rank> strides,
                         value_type init)
    : dims_{dims},
      strides_{strides},
      shared_(new char[strides_[0] * dims_[0]]),
      data_(shared_.get())
  {
    initialize<_Rank>(data_, dims_, strides_, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(std::array<size_t, _Rank> dims,
                         Uninitialized<value_type>)
    : dims_{dims},
      strides_{make_strides<_T>(dims)},
      shared_(new char[strides_[0] * dims_[0]]),
      data_(shared_.get())
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with padding.
  explicit TensorSlowCpu(std::array<size_t, _Rank> dims,
                         std::array<ssize_t, _Rank> strides,
                         Uninitialized<value_type>)
    : dims_{dims},
      strides_{strides},
      shared_(new char[strides_[0] * dims_[0]]),
      data_(shared_.get())
  {}


  /// Constructor ... FIXME
  explicit TensorSlowCpu(std::array<size_t, _Rank> dims,
                         std::array<ssize_t, _Rank> strides,
                         std::shared_ptr<char[]>shared,
                         size_t offset = 0UL)
    : dims_{dims},
      strides_{strides},
      shared_(std::move(shared)),
      data_(shared_.get() + offset)
  {
    // FIXME: assert offset with dims and strides
  }


  // Copy constructor
  // TODO: simple copy; implement reference counted buffers
  TensorSlowCpu(const TensorSlowCpu& other)
    : dims_{other.dims_},
      strides_{other.strides_},
      shared_(other.shared_),
      data_(other.data_)
  { }

  // FIXME: copy?
  /*
  TensorSlowCpu(TensorSlowCpu& other)
    : dims_{other.dims_},
      strides_{other.strides_},
      shared_(other.shared_),
      data_(other.data_)
  { }
  */


  // Move constructor
  TensorSlowCpu(TensorSlowCpu&& other)
    : dims_{other.dims_},
      strides_{other.strides_},
      shared_(std::move(other.shared_)),
      data_(shared_.get())
  {
    other.data_ = nullptr;
  }

  /// Different allocators
  template <TensorFor<TensorSlowCpu> _Tensor>
  TensorSlowCpu(const _Tensor& other)
    : dims_(other.Dims()),
      strides_(other.Strides()),
      shared_(new char[strides_[0] * dims_[0]]),
      data_(shared_.get())
  {
    details::copy<_T, _Rank>(data_, other.Data(), dims_, strides_, other.Strides());
  }


  // Constructors for converting from a tensor operator.
  template <TensorOpFor<TensorSlowCpu> Operator>
  TensorSlowCpu(Operator&& op) : TensorSlowCpu{std::move(op())} {}

  template <TensorOpFor<TensorSlowCpu> Operator>
  TensorSlowCpu(const Operator& op) : TensorSlowCpu{op()} {}

  /// Destructor
  ~TensorSlowCpu()                                        { }

  // FIXME: see below
  TensorSlowCpu& operator=(TensorSlowCpu&& other)
  {
    dims_ = std::move(other.dims_);
    strides_ = std::move(other.strides_);
    shared_ = std::move(other.shared_);
    data_ = shared_.get();
    return *this;
  }

  template<TensorFor<TensorSlowCpu> _Tensor>
  TensorSlowCpu& operator=(const _Tensor& other)
  {
    dims_ = other.Dims();
    strides_ = make_strides<_T>(dims_);
    shared_ = std::shared_ptr<char[]>(new char[strides_[0] * dims_[0]]);
    data_ = shared_.get();
    details::copy<_T, _Rank>(data_, other.Data(), dims_, strides_, other.Strides());
    return *this;
  }


  /// View returns a ... FIXME
  template <size_t _Subrank>
    // FIXME: can make auto again?
  TensorSlowCpu<_T, _Subrank, TensorView{}> View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[_Rank]) &
  {
    return TensorSlowCpu<_T, _Subrank, TensorView{}>(*this, axes, offsets);
  }

  template <size_t _Subrank>
  auto View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[_Rank]) &&
  {
    printf("ERROR create view of temporary!!!\n"); // FIXME
  }

  template <size_t _BroadcastRank = kMaxRank>
  TensorSlowCpu<_T, _BroadcastRank, TensorView{}> Broadcast()
  {
    return TensorSlowCpu<_T, _BroadcastRank, TensorView{}>(*this, _BroadcastRank);
  }

  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _Rank; }

  /// Dims returns the dimensions for the axis.
  const std::array<size_t, _Rank>& Dims() const           { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Data returns a pointer to the data buffer.
  const char* Data() const                                { return data_; }

  /// Data returns a pointer to the data buffer.
  char* Data()                                            { return data_; }


  std::array<size_t, _Rank>         dims_;
  std::array<ssize_t, _Rank>        strides_;
  std::shared_ptr<char[]>           shared_; // FIXME: this migth become a special derived tensor?
  char*                             data_;
};

// FIXME: this might need to become a derived tensor from <_T, _Rank>, and might need to add some additional parameter?
//
/// TensorSlowCpu<_T, 0, 1> is a specialization of a rank-0 tensor.
template <typename _T>
struct TensorSlowCpu<_T, 0>// : TensorBase
{
  using value_type = std::decay_t<_T>;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit TensorSlowCpu(_T init) : array_{init} {}

  explicit TensorSlowCpu(Uninitialized<_T>) {}

  explicit TensorSlowCpu(const std::array<size_t, 0>&, _T init) : array_{init} {}

  explicit TensorSlowCpu(const std::array<size_t, 0>&, Uninitialized<_T>) {}


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 0UL; }

  /// Dims returns the dimensions for the axis.
  const std::array<size_t, 0>& Dims() const               { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 0>& Strides() const           { return {strides_}; }

  /// Data returns a pointer to the data buffer.
  char* Data()                                            { return reinterpret_cast<char*>(array_.data()); }
  const char* Data() const                                { return reinterpret_cast<const char*>(array_.data()); }

  constexpr static std::array<size_t, 0>  dims_{};
  constexpr static std::array<ssize_t, 0> strides_ = {};
  std::array<value_type, 1>               array_;
};


/// TensorSlowCpu<_T, 1, _N> is a specialization of a rank-1 tensor (vector) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _N>
struct TensorSlowCpu<_T, 1, _N>// : TensorBase
{
  using value_type = std::decay_t<_T>;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit TensorSlowCpu(std::initializer_list<_T>&& init)
    : dims_{_N},
      strides_{sizeof(_T)},
      array_(get_array<_T, _N>(std::move(init)))
  {}


  // FIXME: subrank could only be 0 or 1 here?
  // FIXME: return type cannot be const or non-const!??
  template <size_t _Subrank>
  auto View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[1]) const &
  {
    return TensorSlowCpu<_T, 1, TensorView{}>(*this, axes, offsets);
  }

  template <size_t _Subrank>
  auto View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[1]) &
  {
    return TensorSlowCpu<_T, 1, TensorView{}>(*this, axes, offsets);
  }

  template <size_t _Subrank>
  auto View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[1]) const &&
  {
    printf("ERROR create view of temporary!!!\n"); // FIXME
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 1UL; }

  /// Dims returns the dimensions for the axis.
  const std::array<size_t, 1>& Dims() const               { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 1>& Strides() const           { return strides_; }

  /// Data returns a pointer to the data buffer.
  const char* Data() const                                { return reinterpret_cast<const char*>(array_.data()); }


  std::array<size_t, 1>       dims_;
  std::array<ssize_t, 1>      strides_;
  std::array<value_type, _N>  array_;
};


/// TensorSlowCpu<_T, _M, _N> is a specialization of a rank-2 tensor (matrix) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _M, size_t _N>
struct TensorSlowCpu<_T, 2, _M, _N>// : TensorBase
{
  using value_type = std::decay_t<_T>;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit TensorSlowCpu(std::initializer_list<std::initializer_list<_T>>&& init)
    : dims_{_M, _N},
      strides_{ sizeof(_T) * _N, sizeof(_T)},
      array_(get_array<_T, _M, _N>(std::move(init)))
  {}


  // FIXME: subrank could be 0, 1, 2?
  template <size_t _Subrank>
    // FIXME: cannot return const TensorSlow??
  /* const */ auto View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[2]) &
  {
    return TensorSlowCpu<_T, _Subrank, TensorView{}>(*this, axes, offsets);
  }


  template <size_t _Subrank>
  auto View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[2]) &&
  {
    printf("ERROR create view of temporary!!!\n"); // FIXME
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 2UL; }

  /// Dims returns the dimensions for the axis.
  const std::array<size_t, 2>& Dims() const               { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 2>& Strides() const           { return strides_; }

  /// Data returns a pointer to the data buffer.
  const char* Data() const                                { return reinterpret_cast<const char*>(array_.data()); }


  std::array<size_t, 2>     dims_;
  std::array<ssize_t, 2>    strides_;
  std::array<_T, _M * _N>   array_;
};

// FIXME: must be const _T

/// TensorSlowCpu<_T, _C, _M, _N> is a specialization of a rank-3 tensor for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _T, size_t _C, size_t _M, size_t _N>
struct TensorSlowCpu<_T, 3, _C, _M, _N>// : TensorBase
{
  using value_type = std::decay_t<_T>;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit TensorSlowCpu(std::initializer_list<std::initializer_list<std::initializer_list<_T>>>&& init)
    : dims_{_C, _M, _N},
      strides_{ sizeof(_T) * _M * _N, sizeof(_T) * _N, sizeof(_T)},
      array_(get_array<_T, _C, _M, _N>(std::move(init)))
  {}


  // FIXME: Subrank
  template <size_t _Subrank>
  auto View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[3]) &
  {
    return TensorSlowCpu<_T, _Subrank, TensorView{}>(*this, axes, offsets);
  }

  template <size_t _Subrank>
  auto View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[3]) &&
  {
    printf("ERROR create view of temporary!!!\n"); // FIXME:
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 3UL; }

  /// Dims returns the dimensions for the axis.
  const std::array<size_t, 3>& Dims() const               { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 3>& Strides() const           { return strides_; }

  /// Data returns a pointer to the data buffer.
  const char* Data() const                                { return reinterpret_cast<const char*>(array_.data()); }


  std::array<size_t, 3>           dims_;
  std::array<ssize_t, 3>          strides_;
  std::array<_T, _C * _M * _N>    array_;
};


/// TensorSLowCpu<_T, _Rank, MemoryMapped{}> is a tensor for memory-mapped data
template <typename _T, size_t _Rank>
struct TensorSlowCpu<_T, _Rank, MemoryMapped{}>// : TensorBase
{
  using value_type = std::decay_t<_T>;

  explicit TensorSlowCpu() {}

  /// Constructor for a memory-mapped buffer.
  explicit TensorSlowCpu(const MMapArray<_T, _Rank>& arr)
    : dims_(arr.dims_),
      strides_(arr.strides_),
      mmap_(arr.mmap_),
      data_(static_cast<char*>(mmap_->Address()) + arr.offset_)
  {}

  // Constructor for a memory-mapped buffer.
  explicit TensorSlowCpu(MMapArray<_T, _Rank>&& arr)
    : dims_(std::move(arr.dims_)),
      strides_(std::move(arr.strides_)),
      mmap_(std::move(arr.mmap_)),
      data_(static_cast<char*>(mmap_->Address()) + arr.offset_)
  {}


  template <size_t _Subrank>
  auto View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[_Rank]) &
  {
    return TensorSlowCpu<_T, _Subrank, TensorView{}>(*this, axes, offsets);
  }

  template <size_t _Subrank>
  auto View(const size_t(& axes)[_Subrank], const ssize_t(& offsets)[_Rank]) &&
  {
    printf("ERROR create view of temporary!!!\n");  // FIXME
  }


#if 0
  TensorSlowCpu& operator=(TensorSlowCpu&& other)
  {
    dims_ = std::move(other.dims_);
    strides_ = std::move(other.strides_);
    mmap_ = std::move(other.mmap_);
    data_ = other.data_;
    return *this;
  }

  TensorSlowCpu& operator=(const TensorSlowCpu& other)
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

  /// Dims returns the dimensions of the tensor.
  const std::array<size_t, _Rank>& Dims() const           { return dims_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size()                                           { return strides_[0] * dims_[0]; }

  /// Data returns a pointer to the data buffer.
  char* Data() const                                      { return data_; }


  std::array<size_t, _Rank>   dims_;
  std::array<ssize_t, _Rank>  strides_;
  std::shared_ptr<MMap>       mmap_;
  char*                       data_;
};


/// TensorSLowCpu<_T, _Rank, TensorView> is a tensor view
template <typename _T, size_t _Rank>
struct TensorSlowCpu<_T, _Rank, TensorView{}>
{
  using value_type = std::decay_t<_T>;

  template <typename _Tensor, size_t _TensorRank>
  explicit TensorSlowCpu(const _Tensor& tensor, const size_t(& axes)[_Rank], const ssize_t(& offsets)[_TensorRank])
  {
    for (size_t i = 0; i < _Rank; i++)
    {
      dims_[i] = tensor.dims_[axes[i]];
      strides_[i] = tensor.strides_[axes[i]];
    }

    size_t offset = 0;
    for (size_t i = 0; i < _TensorRank; i++)
      offset += offsets[i] * tensor.strides_[i];

    // FIXME handle const, cannot copy into this view!
    data_ = const_cast<char*>(tensor.Data()) + offset;
  }

  // Broadcast
  template <typename _Tensor>
  explicit TensorSlowCpu(const _Tensor& tensor, size_t rank = kMaxRank)
  {
    auto& dims = tensor.Dims();
    auto& strides = tensor.Strides();

    for (size_t i = 0, j = 0; i < rank; i++)
    {
      dims_[i] = i >= tensor.Rank() ? 1 : dims[j];
      strides_[i] = i >= tensor.Rank() ? 0 : strides[j++];
    }
    data_ = const_cast<char*>(tensor.Data());
  }


#if 0
  template<TensorFor<TensorSlowCpu> _View> // requires TensorRank<_View, _Rank>
  explicit TensorSlowCpu(const _View& view)
  {
    printf("O\n");
  }
  template<TensorFor<TensorSlowCpu> _View> // requires TensorRank<_View, _Rank>
  explicit TensorSlowCpu(_View&& view)
  {
    printf("O\n");
  }

  template<TensorViewFor<TensorSlowCpu> _View> // requires TensorRank<_View, _Rank>
  explicit TensorSlowCpu(_View&& view)
  {
    printf("O\n");
  }
#endif

  // TODO: views, check boundaries
  template<TensorViewFor<TensorSlowCpu> _View> requires TensorRank<_View, _Rank>
  auto operator=(const _View& view)
  {
    details::copy<_T, _Rank>(data_, view.Data(), dims_, strides_, view.Strides());
    return *this;
  }

  template<TensorFor<TensorSlowCpu> _View> requires TensorRank<_View, _Rank>
  auto operator=(_View&& view)
  {
    details::copy<_T, _Rank>(data_, view.Data(), dims_, strides_, view.Strides());
    return *this;
  }

  template<TensorViewFor<TensorSlowCpu> _View> requires TensorRank<_View, _Rank>
  auto operator=(const _View& view) const = delete;

  template<TensorViewFor<TensorSlowCpu> _View> requires TensorRank<_View, _Rank>
  auto operator=(_View&&) const = delete;


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _Rank; }

  /// Dims returns the dimensions of the tensor.
  const std::array<size_t, _Rank>& Dims() const           { return dims_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size()                                           { return strides_[0] * dims_[0]; }

  /// Data returns a pointer to the data buffer.
  char* Data() const                                      { return data_; }

  std::array<size_t, _Rank>   dims_;
  std::array<ssize_t, _Rank>  strides_;

  char*                       data_;
};

#if 0
  /// View creates Tensor with a "view" of the data.
  /// This version copies the data to a buffer to create the non-const Tensor
  template <size_t _Subrank>
  auto View(size_t(&& axes)[_Subrank], size_t(&& offsets)[3])
  {
    std::array<size_t, _Subrank> dims;
    std::array<ssize_t, _Subrank> strides;
    for (size_t i = 0; i < _Subrank; i++)
    {
      if (offsets[axes[i]] > dims_[axes[i]])
        throw std::out_of_range("view offset exceeds dimension of tensor");

      dims[i] = dims_[axes[i]] - offsets[axes[i]];
      strides[i] = strides_[axes[i]] + dims_[axes[i]] - dims[axes[i]];
    }

    size_t offset = 0;
    for (size_t i = 0; i < 3; i++)
      offset += offsets[i] * strides_[i];

    return TensorSlowCpu<_T, _Subrank>(dims, strides, Uninitialized<_T>{});
   // reinterpret_cast<const char*>(array_.data()) + offset);
  }

  template <size_t _Subrank>
  auto View(size_t(&& axes)[_Subrank], size_t(&& offsets)[3], size_t(&& dims)[_Subrank])
  {
    std::array<ssize_t, _Subrank> strides;
    for (size_t i = 0; i < _Subrank; i++)
    {
      if (dims[i] > dims_[axes[i]])
        throw std::out_of_range("view dimension exceeds dimension of tensor");

      strides[i] = strides_[axes[i]] + dims_[axes[i]] - dims[i];
    }

    size_t offset = 0;
    for (size_t i = 0; i < 3; i++)
      offset += offsets[i] * strides_[i];

    return TensorViewSlowCpu<const _T, _Subrank>(
        dims, strides, reinterpret_cast<const char*>(array_.data()) + offset);
  }
#endif
#if 0
  auto Row(size_t index)                                  { return View({2}, {index, 0, 0}); }
  auto Col(size_t index)                                  { return View({1}, {0, index, 0}); }
#endif


// CTAD rules

// Tensor{T} -> Rank-0 tensor with a static/local array
template <typename _T>
explicit TensorSlowCpu(_T) -> TensorSlowCpu<_T, 0>;

// Tensor{Uninitailzied<T>} -> Rank-0 tensor with a static/local array
template <typename _T>
explicit TensorSlowCpu(Uninitialized<_T>) -> TensorSlowCpu<_T, 0>;

// Tensor{Ts...} -> Rank-1 tensor with a static/local array (brace-initializer).
template <typename _T, typename... _Ts>
explicit TensorSlowCpu(_T, _Ts...) -> TensorSlowCpu<std::common_type_t<_T, _Ts...>, 1, sizeof...(_Ts)+1>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <typename _T, size_t... _N>
TensorSlowCpu(_T(&&... l)[_N]) -> TensorSlowCpu<_T, 2, sizeof...(_N), std::max({_N...})>;

// Tensor{{{...},...},...} -> Rank-3 tensor with a static/local array (brace-initializer).
template <typename _T, size_t... _M, size_t... _N>
explicit TensorSlowCpu(_T(&&... l)[_M][_N]) -> TensorSlowCpu<_T, 3, sizeof...(_M), std::max({_M...}), std::max({_N...})>;


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


// Tensor(&[], &[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _T, size_t _N>
explicit TensorSlowCpu(const size_t(&)[_N], const ssize_t(&)[_N], _T) -> TensorSlowCpu<_T, _N>;

// Tensor(&[], &[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _T, size_t _N>
explicit TensorSlowCpu(const size_t(&)[_N], const ssize_t(&)[_N], Uninitialized<_T>) -> TensorSlowCpu<_T, _N>;

// Tensor(&&[], &&[], _T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _T, size_t _N>
explicit TensorSlowCpu(size_t(&&)[_N], ssize_t(&&)[_N], _T) -> TensorSlowCpu<_T, _N>;

// Tensor(&&[], &&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _T, size_t _N>
explicit TensorSlowCpu(size_t(&&)[_N], ssize_t(&&)[_N], Uninitialized<_T>) -> TensorSlowCpu<_T, _N>;


// Tensor(&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _T, size_t _N>
explicit TensorSlowCpu(const size_t(&)[_N], _T) -> TensorSlowCpu<_T, _N>;

// Tensor(&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _T, size_t _N>
explicit TensorSlowCpu(const size_t(&)[_N], Uninitialized<_T>) -> TensorSlowCpu<_T, _N>;

// Tensor(&&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename _T, size_t _N>
explicit TensorSlowCpu(const size_t(&&)[_N], _T) -> TensorSlowCpu<_T, _N>;

// Tensor(&&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _T, size_t _N>
explicit TensorSlowCpu(const size_t(&&)[_N], Uninitialized<_T>) -> TensorSlowCpu<_T, _N>;


// Tensor(array, T)
template <typename _T, size_t _N>
TensorSlowCpu(std::array<size_t, _N>, _T) -> TensorSlowCpu<_T, _N>;

// Tensor(array, array, T)
template <typename _T, size_t _N>
explicit TensorSlowCpu(std::array<size_t, _N>, std::array<ssize_t, _N>, _T) -> TensorSlowCpu<_T, _N>;

// Tensor(array, Uninitialized<T>)
template <typename _T, size_t _N>
explicit TensorSlowCpu(std::array<size_t, _N>, Uninitialized<_T>) -> TensorSlowCpu<_T, _N>;

// Tensor(array, array, Uninitialized<T>)
template <typename _T, size_t _N>
explicit TensorSlowCpu(std::array<size_t, _N>, std::array<ssize_t, _N>, Uninitialized<_T>) -> TensorSlowCpu<_T, _N>;


// Tensor<mmap, dim, strides> -> Rank-N tensor for a memory mapped buffer
template <typename _T, size_t _N>
explicit TensorSlowCpu(MMapArray<_T, _N>) -> TensorSlowCpu<_T, _N, MemoryMapped{}>;

// Tensor(Tensor...) FIXME: needed? will it work??
template <typename _T, size_t _N>
explicit TensorSlowCpu(TensorSlowCpu<_T, 1, _N>&) -> TensorSlowCpu<_T, _N>;

template <typename _T, size_t _N, size_t _M>
explicit TensorSlowCpu(TensorSlowCpu<_T, 2, _N, _M>&) -> TensorSlowCpu<_T, _N * _M>;

template <typename _T, size_t _C, size_t _N, size_t _M>
explicit TensorSlowCpu(TensorSlowCpu<_T, 3, _C, _N, _M>&) -> TensorSlowCpu<_T, _C * _N * _M>;


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

#if 0
// FIXME: remove

    std::array<size_t, _Subrank> dims;
    std::transform(axes.cbegin(), axes.cend(), std::back_inserter(dims),
                   [=](size_t axes) { return dims_[axes]; });

    std::array<size_t, _Subrank> strides;
    std::transform(axes.cbegin(), axes.cend(), std::back_inserter(strides),
                   [=](size_t axes) { return strides_[axes]; });

    //std::shared_ptr<_T[]> shared(reinterpret_cast<_T*>(new char[strides[0] * dims_[0]]));
    details::copy<_T, _Subrank>(reinterpret_cast<char*>(other.Data(), shared.get()),
        reinterpret_cast<const char*>(array_.data()) + offset, dims, strides, strides_src);
    return TensorSlowCpu<_T, _Subrank>(std::move(dims), std::move(strides), Uninitialized<_T>{});
#if 0
    dims_ = other.dims_;
    strides_ = other.strides_;
    shared_ = other.shared_;
    data_ = shared_.get();
    return *this;
#endif
 #endif
