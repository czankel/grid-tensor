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
#include <bitset>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <span>

#include <grid/tensor/mmap.h>

#include "copy.h"

namespace grid {

/// TensorSlowCpu is an unoptimized tensor implementatoin for the CPU.
template <typename T, size_t, typename = StandardAllocator<T>> class TensorSlowCpu;

/// TensorSlowCpu<T, _Rank> is a specialization of TensorSlowCpu for a dynamically allocated buffer.
///
/// Note that this is also the Tensor used for any TensorOp result.
/// TODO: see if constructors can be combined using implicit conversion
template <typename T, size_t _Rank>
class TensorSlowCpu<T, _Rank, StandardAllocator<T>>
{
 public:
  using value_type = T;
  using memory_type = StandardAllocator<T>;
  using pointer_type = T*;
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
  TensorSlowCpu() = default;

  /// Destructor
  ~TensorSlowCpu()
  {
    if (data_ != nullptr)
      delete[] data_;
  }


  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  explicit TensorSlowCpu(size_t dim, value_type init)
    : dims_{dim},
      strides_{make_strides<T>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(size_t dim, Uninitialized<value_type>)
    : dims_{dim},
      strides_{make_strides<T>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit TensorSlowCpu(size_t dim_m, int dim_n, value_type init)
    : dims_{dim_m, dim_n},
      strides_{make_strides<T>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(size_t dim_m, int dim_n, Uninitialized<value_type>)
    : dims_{dim_m, dim_n},
      strides_{make_strides<T>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims, value_type init)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_{make_strides<T>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims, Uninitialized<value_type>)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_{make_strides<T>(dims_)},
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with strides.
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims,
                         std::initializer_list<ssize_t>&& strides,
                         value_type init)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_(get_array<ssize_t, _Rank>(std::move(strides))),
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with strides.
  explicit TensorSlowCpu(std::initializer_list<size_t>&& dims,
                         std::initializer_list<ssize_t>&& strides,
                         Uninitialized<value_type>)
    : dims_(get_array<size_t, _Rank>(std::move(dims))),
      strides_(get_array<ssize_t, _Rank>(std::move(strides))),
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {}

  // FIXME: can this be combined with initializer-list?
  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit TensorSlowCpu(const size_t(&dim)[_Rank], const ssize_t(&stride)[_Rank], value_type init)
    : dims_(get_array<size_t, _Rank>(dim)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {
    initialize(data_, std::span{dims_}, std::span{strides_}, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  explicit TensorSlowCpu(const size_t(&dim)[_Rank], const ssize_t(&stride)[_Rank], Uninitialized<T>)
    : dims_(get_array<size_t, _Rank>(dim)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer.
  explicit TensorSlowCpu(std::array<size_t, _Rank> dims,
                         value_type init)
    : dims_(dims),
      strides_(make_strides<T>(dims)),
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {
    initialize<_Rank>(data_, dims_, strides_, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with padding.
  explicit TensorSlowCpu(std::array<size_t, _Rank> dims,
                         std::array<ssize_t, _Rank> strides,
                         value_type init)
    : dims_{dims},
      strides_{strides},
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {
    initialize<_Rank>(data_, dims_, strides_, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer.
  explicit TensorSlowCpu(std::array<size_t, _Rank> dims,
                         Uninitialized<value_type>)
    : dims_{dims},
      strides_{make_strides<T>(dims)},
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with padding.
  explicit TensorSlowCpu(std::array<size_t, _Rank> dims,
                         std::array<ssize_t, _Rank> strides,
                         Uninitialized<value_type>)
    : dims_{dims},
      strides_{strides},
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {}


  // Copy constructor
  TensorSlowCpu(const TensorSlowCpu& other)
    : dims_{other.Dimensions()},
      strides_{other.Strides()},
      size_(other.Size()),
      data_(new  char[size_])
  {
    details::copy<T, _Rank>(data_, other.Data(), dims_, strides_, other.Strides());
  }

  // Move constructor
  TensorSlowCpu(TensorSlowCpu&& other)
    : dims_{other.dims_},
      strides_{other.strides_},
      size_(other.size_),
      data_(std::move(other.data_))
  {
    other.data_ = nullptr;
  }


  /// Different allocators FIXME what's this?
  template <TensorFor<TensorSlowCpu> Tensor>
  TensorSlowCpu(const Tensor& other)
    : dims_(other.Dimensions()),
      strides_(other.Strides()),
      size_(dims_[0] * strides_[0]),
      data_(new  char[size_])
  {
    details::copy<T, _Rank>(data_, other.Data(), dims_, strides_, other.Strides());
  }


  // Constructors for converting from a tensor operator.
  template <TensorOpFor<TensorSlowCpu> Operator>
  TensorSlowCpu(Operator&& functor) : TensorSlowCpu{std::move(functor())} {}

  template <TensorOpFor<TensorSlowCpu> Operator>
  TensorSlowCpu(const Operator& functor) : TensorSlowCpu{functor()} {}

#if 0
  // FIXME: see below
  TensorSlowCpu& operator=(TensorSlowCpu&& other)
  {
    dims_ = std::move(other.dims_);
    strides_ = std::move(other.strides_);
    size_ = other.size_;

    if (data_ != nullptr)
      delete[] data_;
    data_ = std::move(other.data_);
    other.data_ = nullptr;

    return *this;
  }
#endif
  // FIXME: like this, but must treat anything else than bare-bone tensor as "Copy"
#if 0
  template<TensorFor<TensorSlowCpu> Tensor>
  TensorSlowCpu& operator=(Tensor&& other)
#endif

  template<TensorFor<TensorSlowCpu> Tensor>
  TensorSlowCpu& operator=(const Tensor& other)
  {
    dims_ = other.Dimensions();
    strides_ = make_strides<T>(dims_);
    size_ = strides_[0] * dims_[0];
    if (data_ != nullptr)
      delete[] data_;
    data_ = new char[size_];
    details::copy<T, _Rank>(data_, other.Data(), dims_, strides_, other.Strides());
    return *this;
  }


  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank] = {0}) const &
  {
    return TensorSlowCpu<T, _ViewRank, TensorView<T>>(*this, axes, offsets);
  }

  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank]) const && = delete;


  /// Rank returns the rank of the tensor.
  //constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, _Rank>& Dimensions() const     { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  const char* Data() const                                { return data_; }

  /// Data returns a pointer to the data buffer.
  char* Data()                                            { return data_; }


  std::array<size_t, _Rank>         dims_;
  std::array<ssize_t, _Rank>        strides_;
  size_t                            size_;
  char*                             data_;
};


// FIXME: this might need to become a derived tensor from <T, _Rank>, and might need to add some additional parameter?
//
/// TensorSlowCpu<T, 0, 1> is a specialization of a rank-0 tensor.
template <typename T>
class TensorSlowCpu<T, 0>
{
 public:
  using value_type = std::decay_t<T>;
  constexpr static size_t rank = 0UL;;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit TensorSlowCpu(T init) : array_{init} {}

  explicit TensorSlowCpu(Uninitialized<T>) {}

  explicit TensorSlowCpu(const std::array<size_t, 0>&, T init) : array_{init} {}

  explicit TensorSlowCpu(const std::array<size_t, 0>&, Uninitialized<T>) {}


  /// Rank returns the rank of the tensor.
  //constexpr static size_t Rank()                          { return 0UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 0>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 0>& Strides() const           { return {strides_}; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type); }

  /// Data returns a pointer to the data buffer.
  char* Data()                                            { return reinterpret_cast<char*>(array_.data()); }
  const char* Data() const                                { return reinterpret_cast<const char*>(array_.data()); }

  constexpr static std::array<size_t, 0>  dims_{};
  constexpr static std::array<ssize_t, 0> strides_ = {};
  std::array<value_type, 1>               array_;
};


/// TensorSlowCpu<T, 1, _N> is a specialization of a rank-1 tensor (vector) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename T, size_t _N>
class TensorSlowCpu<T, 1, StaticAllocator<T, _N>>
{
 public:
  using value_type = std::decay_t<T>;
  constexpr static size_t rank = 1UL;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit TensorSlowCpu(std::initializer_list<T>&& init)
    : dims_{_N},
      strides_{sizeof(T)},
      array_(get_array<T, _N>(std::move(init)))
  {}

// FIXME: can this be const always or need two options?
  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[1]) const &
  {
    return TensorSlowCpu<T, _ViewRank, TensorView<T>>(*this, axes, offsets);
  }

  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[1]) && = delete;


  /// Rank returns the rank of the tensor.
  //constexpr static size_t Rank()                          { return 1UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 1>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 1>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type) * _N; }

  /// Data returns a pointer to the data buffer.
  const char* Data() const                                { return reinterpret_cast<const char*>(array_.data()); }


  std::array<size_t, 1>       dims_;
  std::array<ssize_t, 1>      strides_;
  std::array<value_type, _N>  array_;
};


/// TensorSlowCpu<T, _M, _N> is a specialization of a rank-2 tensor (matrix) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename T, size_t _M, size_t _N>
class TensorSlowCpu<T, 2, StaticAllocator<T, _M, _N>>
{
 public:
  using value_type = std::decay_t<T>;
  constexpr static size_t rank = 2UL;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit TensorSlowCpu(std::initializer_list<std::initializer_list<T>>&& init)
    : dims_{_M, _N},
      strides_{ sizeof(T) * _N, sizeof(T)},
      array_(get_array<T, _M, _N>(std::move(init)))
  {}

  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[2]) const &
  {
    return TensorSlowCpu<T, _ViewRank, TensorView<T>>(*this, axes, offsets);
  }

  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[2]) && = delete;


  /// Rank returns the rank of the tensor.
  //constexpr static size_t Rank()                          { return 2UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 2>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 2>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type) * _M * _N; }

  /// Data returns a pointer to the data buffer.
  const char* Data() const                                { return reinterpret_cast<const char*>(array_.data()); }


  std::array<size_t, 2>     dims_;
  std::array<ssize_t, 2>    strides_;
  std::array<T, _M * _N>   array_;
};


/// TensorSlowCpu<T, _C, _M, _N> is a specialization of a rank-3 tensor for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename T, size_t _C, size_t _M, size_t _N>
class TensorSlowCpu<T, 3, StaticAllocator<T, _C, _M, _N>>
{
 public:
  using value_type = std::decay_t<T>;
  constexpr static size_t rank = 3UL;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit TensorSlowCpu(std::initializer_list<std::initializer_list<std::initializer_list<T>>>&& init)
    : dims_{_C, _M, _N},
      strides_{ sizeof(T) * _M * _N, sizeof(T) * _N, sizeof(T)},
      array_(get_array<T, _C, _M, _N>(std::move(init)))
  {}

  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[3]) const &
  {
    return TensorSlowCpu<T, _ViewRank, TensorView<T>>(*this, axes, offsets);
  }

  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[3]) && = delete;


  /// Rank returns the rank of the tensor.
  //constexpr static size_t Rank()                          { return 3UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 3>& Dimensions() const         { return dims_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 3>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type) * _C * _M * _N; }

  /// Data returns a pointer to the data buffer.
  const char* Data() const                                { return reinterpret_cast<const char*>(array_.data()); }


  std::array<size_t, 3>           dims_;
  std::array<ssize_t, 3>          strides_;
  std::array<T, _C * _M * _N>    array_; // FIXME: C, M, N not really needed....
};


/// TensorSLowCpu<T, _Rank, MemoryMapped{}> is a tensor for memory-mapped data
template <typename T, size_t _Rank>
class TensorSlowCpu<T, _Rank, MemoryMapped<T>>
{
 public:
  using value_type = std::decay_t<T>;
  constexpr static size_t rank = _Rank;

  explicit TensorSlowCpu() {}

  /// Constructor for a memory-mapped buffer.
  explicit TensorSlowCpu(const MMapArray<T, _Rank>& arr)
    : //size_(strides_[0] * dims_[0]),
      dims_(arr.dims_),
      strides_(arr.strides_),
      mmap_(arr.mmap_),
      data_(static_cast<char*>(mmap_->Address()) + arr.offset_)
  {}

  // Constructor for a memory-mapped buffer.
  explicit TensorSlowCpu(MMapArray<T, _Rank>&& arr)
    : dims_(std::move(arr.dims_)),
      strides_(std::move(arr.strides_)),
      mmap_(std::move(arr.mmap_)),
      data_(static_cast<char*>(mmap_->Address()) + arr.offset_)
  {}

  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank]) const &
  {
    return TensorSlowCpu<T, _ViewRank, TensorView<T>>(*this, axes, offsets);
  }

  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank]) && = delete;


#if 0
  // FIXME: support operator= on memorymapped? need two versions, RO and RW?
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
  //constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, _Rank>& Dimensions() const     { return dims_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size() const                                     { return mmap_->Size(); }

  /// Data returns a pointer to the data buffer.
  char* Data() const                                      { return data_; }


  size_t                      size_;
  std::array<size_t, _Rank>   dims_;
  std::array<ssize_t, _Rank>  strides_;
  std::shared_ptr<MMap>       mmap_;
  char*                       data_;
};


/// TensorSLowCpu<T, _Rank, TensorView> is a tensor view
///
/// A view cannot be created from a temporary rval; it will return a tensor
template <typename T, size_t _Rank>
class TensorSlowCpu<T, _Rank, TensorView<T>>
{
 public:
  using value_type = std::decay_t<T>;
  constexpr static size_t rank = _Rank;

  template <typename Tensor, size_t TensorRank>
  explicit TensorSlowCpu(const Tensor& tensor, const ssize_t(& axes)[_Rank])
    : TensorSlowCpu(tensor, axes, { 0 })
  {}

  template <typename Tensor, size_t TensorRank>
  explicit TensorSlowCpu(const Tensor& tensor, const ssize_t(& axes)[_Rank], const ssize_t(& offsets)[TensorRank])
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
        strides_[i] = 0; // i < static_cast<ssize_t>(_Rank)-1 ? strides_[i+1] : sizeof(T);
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

    data_ = const_cast<char*>(tensor.Data()) + offset;
  }


  // TODO: views, check boundaries
  template<TensorViewFor<TensorSlowCpu> _View> requires TensorRank<_View, _Rank>
  auto operator=(const _View& view)
  {
    details::copy<T, _Rank>(data_, view.Data(), dims_, strides_, view.Strides());
    return *this;
  }

  template<TensorFor<TensorSlowCpu> _View> requires TensorRank<_View, _Rank>
  auto operator=(_View&& view)
  {
    details::copy<T, _Rank>(data_, view.Data(), dims_, strides_, view.Strides());
    return *this;
  }

  template<TensorViewFor<TensorSlowCpu> _View> requires TensorRank<_View, _Rank>
  auto operator=(const _View& view) const = delete;

  template<TensorViewFor<TensorSlowCpu> _View> requires TensorRank<_View, _Rank>
  auto operator=(_View&&) const = delete;


  /// Rank returns the rank of the tensor.
  //constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, _Rank>& Dimensions() const     { return dims_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size()                                           { return strides_[0] * dims_[0]; } // FIXME: wrong

  /// Data returns a pointer to the data buffer.
  char* Data() const                                      { return data_; }

  std::array<size_t, _Rank>   dims_;
  std::array<ssize_t, _Rank>  strides_;

  char*                       data_;
};

#if 0
// FIXME: need to 'amend' with rank-2 0s...
  auto Row(size_t index)                                  { return View({1}, {index, ....}); }
  auto Col(size_t index)                                  { return View({0}, {0, index, 0}); }
#endif


// CTAD rules

// Tensor{T} -> Rank-0 tensor with a static/local array
template <typename T>
explicit TensorSlowCpu(T) -> TensorSlowCpu<T, 0>;

// Tensor{Uninitailzied<T>} -> Rank-0 tensor with a static/local array
template <typename T>
explicit TensorSlowCpu(Uninitialized<T>) -> TensorSlowCpu<T, 0>;

// Tensor{Ts...} -> Rank-1 tensor with a static/local array (brace-initializer).
template <typename T, typename... Ts>
explicit TensorSlowCpu(T, Ts...) -> TensorSlowCpu<std::common_type_t<T, Ts...>, 1, StaticAllocator<T, sizeof...(Ts)+1>>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <typename T, size_t... _N>
TensorSlowCpu(T(&&... l)[_N]) -> TensorSlowCpu<T, 2, StaticAllocator<T, sizeof...(_N), std::max({_N...})>>;

// Tensor{{{...},...},...} -> Rank-3 tensor with a static/local array (brace-initializer).
template <typename T, size_t... _M, size_t... _N>
explicit TensorSlowCpu(T(&&... l)[_M][_N]) -> TensorSlowCpu<T, 3, StaticAllocator<T, sizeof...(_M), std::max({_M...}), std::max({_N...})>>;


// Tensor(uint, T) -> Rank-1 tensor with a dynamically allocated buffer.
template <typename T>
explicit TensorSlowCpu(size_t, T) -> TensorSlowCpu<T, 1>;

// Tensor(uint, Uninitialized<T>) -> Rank-1 tensor with a dynamically allocated uninitialized buffer.
template <typename T>
explicit TensorSlowCpu(size_t, Uninitialized<T>) -> TensorSlowCpu<T, 1>;

// Tensor(uint, uint, T) -> Rank-2 tensor with a dynamically allocated buffer.
template <typename T>
explicit TensorSlowCpu(size_t, size_t, T) -> TensorSlowCpu<T, 2>;

// Tensor(uint, Uninitialized<T>) -> Rank-2 tensor with a dynamically allocated uninitialized buffer.
template <typename T>
explicit TensorSlowCpu(size_t, size_t, Uninitialized<T>) -> TensorSlowCpu<T, 2>;


// Tensor(&[], &[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename T, size_t _N>
explicit TensorSlowCpu(const size_t(&)[_N], const ssize_t(&)[_N], T) -> TensorSlowCpu<T, _N>;

// Tensor(&[], &[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename T, size_t _N>
explicit TensorSlowCpu(const size_t(&)[_N], const ssize_t(&)[_N], Uninitialized<T>) -> TensorSlowCpu<T, _N>;

// Tensor(&&[], &&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename T, size_t _N>
explicit TensorSlowCpu(size_t(&&)[_N], ssize_t(&&)[_N], T) -> TensorSlowCpu<T, _N>;

// Tensor(&&[], &&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename T, size_t _N>
explicit TensorSlowCpu(size_t(&&)[_N], ssize_t(&&)[_N], Uninitialized<T>) -> TensorSlowCpu<T, _N>;


// Tensor(&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename T, size_t _N>
explicit TensorSlowCpu(const size_t(&)[_N], T) -> TensorSlowCpu<T, _N>;

// Tensor(&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename T, size_t _N>
explicit TensorSlowCpu(const size_t(&)[_N], Uninitialized<T>) -> TensorSlowCpu<T, _N>;

// Tensor(&&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <typename T, size_t _N>
explicit TensorSlowCpu(const size_t(&&)[_N], T) -> TensorSlowCpu<T, _N>;

// Tensor(&&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename T, size_t _N>
explicit TensorSlowCpu(const size_t(&&)[_N], Uninitialized<T>) -> TensorSlowCpu<T, _N>;


// Tensor(array, T)
template <typename T, size_t _N>
TensorSlowCpu(std::array<size_t, _N>, T) -> TensorSlowCpu<T, _N>;

// Tensor(array, array, T)
template <typename T, size_t _N>
explicit TensorSlowCpu(std::array<size_t, _N>, std::array<ssize_t, _N>, T) -> TensorSlowCpu<T, _N>;

// Tensor(array, Uninitialized<T>)
template <typename T, size_t _N>
explicit TensorSlowCpu(std::array<size_t, _N>, Uninitialized<T>) -> TensorSlowCpu<T, _N>;

// Tensor(array, array, Uninitialized<T>)
template <typename T, size_t _N>
explicit TensorSlowCpu(std::array<size_t, _N>, std::array<ssize_t, _N>, Uninitialized<T>) -> TensorSlowCpu<T, _N>;


// Tensor<mmap, dim, strides> -> Rank-N tensor for a memory mapped buffer
template <typename T, size_t _N>
explicit TensorSlowCpu(MMapArray<T, _N>) -> TensorSlowCpu<T, _N, MemoryMapped<T>>;

// Tensor(Tensor...) FIXME: needed? will it work??
template <typename T, size_t _N>
explicit TensorSlowCpu(TensorSlowCpu<T, 1, StaticAllocator<T, _N>>&) -> TensorSlowCpu<T, _N>;

template <typename T, size_t _N, size_t _M>
explicit TensorSlowCpu(TensorSlowCpu<T, 2, StaticAllocator<T, _N, _M>>&) -> TensorSlowCpu<T, _N * _M>;

template <typename T, size_t _C, size_t _N, size_t _M>
explicit TensorSlowCpu(TensorSlowCpu<T, 3, StaticAllocator<T, _C, _N, _M>>&) -> TensorSlowCpu<T, _C * _N * _M>;


// TensorOp -> Tensor (move)
template <template <template <typename, size_t, typename> typename, typename, size_t, typename...> typename TensorOp,
          template <typename, size_t, typename> typename TensorRT, typename T, size_t _Rank, typename... Tensors>
TensorSlowCpu(TensorOp<TensorRT, T, _Rank, Tensors...>&&) -> TensorSlowCpu<T, _Rank>;

// TensorOp -> Tensor (copy)
template <template <template <typename, size_t, typename> typename, typename, size_t, typename...> typename TensorOp,
          template <typename, size_t, typename> typename TensorRT, typename T, size_t _Rank, typename... Tensors>
TensorSlowCpu(const TensorOp<TensorRT,T,  _Rank, Tensors...>&) -> TensorSlowCpu<T, _Rank>;


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
 #endif
