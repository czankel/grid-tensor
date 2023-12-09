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
#include <bitset>
#include <initializer_list>

#include "../array.h"
#include "../tensor_parameters.h"

#include "copy.h"

namespace grid {

template <PrimitiveTensor, size_t> class TensorView;

/// Tensor provides an non-optimized base implementation of tensors.
/// Note that the implementation implicitly requires that the buffer and strides are aligned to the value type.
template <typename _Tp, size_t _Rank, typename... _Allocator> class Tensor;

/// Tensor is the default tensor using the standard std::allocator
template <typename _Tp, size_t _Rank>
class Tensor<_Tp, _Rank>
{
  template <PrimitiveTensor T, size_t R> friend class TensorView;

 public:
  using value_type = _Tp;
  using allocator_type = std::allocator<_Tp>;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = _Rank;

 private:
  inline void
  initialize(pointer ptr,
             std::span<size_t, 1> dimensions,
             std::span<ssize_t, 1> strides,
             value_type init)
  {
    for (size_t i = 0; i < dimensions[0]; i++, ptr = ptr + strides[0]/sizeof(value_type))
      *ptr = init;
  }

  template <size_t _N>
  inline void
  initialize(pointer ptr,
             std::span<size_t, _N> dimensions,
             std::span<ssize_t, _N> strides,
             value_type init)
  {
    for (size_t i = 0; i < dimensions[0]; i++, ptr += strides[0]/sizeof(value_type))
      initialize(ptr,
                 std::span<size_t, _N - 1>(dimensions.begin() + 1, dimensions.end()),
                 std::span<ssize_t, _N - 1>(strides.begin() + 1, strides.end()),
                 init);
  }

 public:
  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  explicit Tensor(size_t dimension, value_type init)
    : dimensions_{dimension},
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dimensions_}, std::span{strides_}, init);
  }

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dimension, Uninitialized<value_type>)
    : dimensions_{dimension},
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit Tensor(size_t dim_m, int dim_n, value_type init)
    : dimensions_{dim_m, dim_n},
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dimensions_}, std::span{strides_}, init);
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dim_m, int dim_n, Uninitialized<value_type>)
    : dimensions_{dim_m, dim_n},
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(std::initializer_list<size_t>&& dimensions, value_type init)
    : dimensions_(get_array<size_t, _Rank>(std::move(dimensions))),
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dimensions_}, std::span{strides_}, init);
  }


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(std::initializer_list<size_t>&& dimensions, Uninitialized<value_type>)
    : dimensions_(get_array<size_t, _Rank>(std::move(dimensions))),
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with strides.
  explicit Tensor(std::initializer_list<size_t>&& dimensions,
                  std::initializer_list<ssize_t>&& strides,
                  value_type init)
    : dimensions_(get_array<size_t, _Rank>(std::move(dimensions))),
      strides_(get_array<ssize_t, _Rank>(std::move(strides))),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dimensions_}, std::span{strides_}, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with strides.
  explicit Tensor(std::initializer_list<size_t>&& dimensions,
                  std::initializer_list<ssize_t>&& strides,
                  Uninitialized<value_type>)
    : dimensions_(get_array<size_t, _Rank>(std::move(dimensions))),
      strides_(get_array<ssize_t, _Rank>(std::move(strides))),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(const size_t(&dimension)[_Rank], const ssize_t(&stride)[_Rank], value_type init)
    : dimensions_(get_array<size_t, _Rank>(dimension)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dimensions_}, std::span{strides_}, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  explicit Tensor(const size_t(&dimension)[_Rank], const ssize_t(&stride)[_Rank], Uninitialized<value_type>)
    : dimensions_(get_array<size_t, _Rank>(dimension)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer.
  explicit Tensor(std::array<size_t, _Rank> dimensions, value_type init)
    : dimensions_(dimensions),
      strides_(make_strides<value_type>(dimensions)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize<_Rank>(data_, dimensions_, strides_, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with padding.
  explicit Tensor(std::array<size_t, _Rank> dimensions,
                  std::array<ssize_t, _Rank> strides,
                  value_type init)
    : dimensions_{dimensions},
      strides_{strides},
      size_(get_buffer_size(dimensions_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize<_Rank>(data_, dimensions_, strides_, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer.
  /// Note: assumes strides are type-aligned.
  explicit Tensor(std::array<size_t, _Rank> dimensions, Uninitialized<value_type>)
    : dimensions_{dimensions},
      strides_{make_strides<value_type>(dimensions)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}


  /// Constructor from a 'trivially copyable' tensor.
  /// Note: assumes strides are type-aligned.
  template <PrimitiveTensor _Tensor>
  Tensor(const _Tensor& other)
    : dimensions_(other.Dimensions()),
      strides_(other.Strides()),
      size_(other.Size()),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    copy<value_type, _Rank>(data_, other.Data(), dimensions_, strides_, other.Strides());
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with padding.
  explicit Tensor(std::array<size_t, _Rank> dimensions,
                  std::array<ssize_t, _Rank> strides,
                  Uninitialized<value_type>)
    : dimensions_{dimensions},
      strides_{strides},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {}

  // Constructors for converting from a tensor operator.
  template <AnyOperator Operator>
  Tensor(Operator&& functor) : Tensor{std::move(functor())} {};

  template <AnyOperator Operator>
  Tensor(const Operator& functor) : Tensor{functor()} {};


  /// Copy constructor
  Tensor(const Tensor& other)
    : dimensions_{other.Dimensions()},
      strides_{other.Strides()},
      size_(other.Size()),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    copy<value_type, _Rank>(data_, other.Data(), dimensions_, strides_, other.Strides());
  }

  /// Move constructor
  Tensor(Tensor&& other)
    : dimensions_{std::move(other.dimensions_)},
      strides_{std::move(other.strides_)},
      size_(std::move(other.size_)),
      data_(std::move(other.data_))
  {
    other.data_ = nullptr;
  }

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
    dimensions_ = other.Dimensions();
    strides_ = make_strides<value_type>(dimensions_);
    size_ = strides_[0] * dimensions_[0];
    if (data_ != nullptr)
      delete[] data_;
    data_ = new value_type[size_ / sizeof(value_type)];
    copy<value_type, _Rank>(data_, other.Data(), dimensions_, strides_, other.Strides());
    return *this;
  }

  /// Move-assign is only supported from the same type
  Tensor& operator=(Tensor&& other)
  {
    dimensions_ = other.Dimensions();
    strides_ = make_strides<value_type>(dimensions_);
    size_ = strides_[0] * dimensions_[0];
    if (data_ != nullptr)
      delete[] data_;
    data_ = other.Data();
    return *this;
  }

  /// Operator assign
  template <AnyOperator _Operator>
  Tensor& operator=(_Operator&& oper)
  {
    return operator=(std::forward<_Operator>(oper)());
  }


  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank] = {0})
  {
    return TensorView(*this, axes, offsets);
  }

  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank] = {0}) const
  {
    return TensorView(*this, axes, offsets);
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, _Rank>& Dimensions() const     { return dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return data_; }
  const_pointer Data() const                              { return data_; }

 private:
  std::array<size_t, _Rank>         dimensions_;
  std::array<ssize_t, _Rank>        strides_;
  size_t                            size_;
  pointer                           data_;
};

/// Tensor<T, 0, 1> is a specialization of a rank-0 tensor.
template <typename _Tp>
class Tensor<_Tp, 0>
{
 public:
  using value_type = _Tp;
  using allocator_type = StaticAllocator<1>;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = 0UL;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit Tensor(value_type init) : data_{init} {}

  explicit Tensor(Uninitialized<value_type>) {}

  explicit Tensor(const std::array<size_t, 0>&, value_type init) : data_{init} {}

  explicit Tensor(const std::array<size_t, 0>&, Uninitialized<value_type>) {}

  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 0UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 0>& Dimensions() const         { return dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 0>& Strides() const           { return {strides_}; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type); }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return &data_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return &data_; }

 private:
  constexpr static std::array<size_t, 0>  dimensions_{};
  constexpr static std::array<ssize_t, 0> strides_ = {};
  value_type                              data_;
};


/// Tensor<T, 1, N> is a specialization of a rank-1 tensor (vector) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _Tp, size_t _N>
class Tensor<_Tp, 1, StaticAllocator<_N>>
{
 public:
  using value_type = _Tp;
  using allocator_type = StaticAllocator<_N>;
  using pointer = const _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = 1UL;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit Tensor(std::initializer_list<value_type>&& init)
    : dimensions_{_N},
      strides_{sizeof(value_type)},
      array_(get_array<value_type, _N>(std::move(init)))
  {}


  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[1]) const &
  {
    return TensorView(*this, axes, offsets);
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 1UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 1>& Dimensions() const         { return dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 1>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type) * _N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }

 private:
  std::array<size_t, 1>       dimensions_;
  std::array<ssize_t, 1>      strides_;
  std::array<value_type, _N>  array_;
};


/// Tensor<T, M, N> is a specialization of a rank-2 tensor (matrix) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _Tp, size_t _M, size_t _N>
class Tensor<_Tp, 2, StaticAllocator<_M, _N>>
{
 public:
  using value_type = _Tp;
  using pointer = const _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = 2UL;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit Tensor(std::initializer_list<std::initializer_list<value_type>>&& init)
    : dimensions_{_M, _N},
      strides_{ sizeof(value_type) * _N, sizeof(value_type)},
      array_(get_array<value_type, _M, _N>(std::move(init)))
  {}


  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[2]) const &
  {
    return TensorView(*this, axes, offsets);
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return 2UL; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, 2>& Dimensions() const         { return dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 2>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type) * _M * _N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }

 private:
  std::array<size_t, 2>           dimensions_;
  std::array<ssize_t, 2>          strides_;
  std::array<value_type, _M * _N> array_;
};


/// Tensor<T, C, M, N> is a specialization of a rank-3 tensor for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename _Tp, size_t _C, size_t _M, size_t _N>
class Tensor<_Tp, 3, StaticAllocator<_C, _M, _N>>
{
 public:
  using value_type = _Tp;
  using allocator_type = StaticAllocator<_C, _M, _N>;
  using pointer = const _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = 3UL;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit Tensor(std::initializer_list<std::initializer_list<std::initializer_list<value_type>>>&& init)
    : dimensions_{_C, _M, _N},
      strides_{ sizeof(value_type) * _M * _N, sizeof(value_type) * _N, sizeof(value_type)},
      array_(get_array<value_type, _C, _M, _N>(std::move(init)))
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
  const std::array<size_t, 3>& Dimensions() const         { return dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, 3>& Strides() const           { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type) * _C * _M * _N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }


  std::array<size_t, 3>                 dimensions_;
  std::array<ssize_t, 3>                strides_;
  std::array<value_type, _C * _M * _N>  array_;
};


/// Tensor<T, Rank, NoAllocator> is a tensor for an externally managed buffer
template <typename _Tp, size_t _Rank>
class Tensor<_Tp, _Rank, NoAllocator>
{
 public:
  using value_type = _Tp;
  using allocator_type = NoAllocator;
  using pointer = const _Tp*;
  using const_pointer = const _Tp*;
  constexpr static size_t rank = _Rank;

  explicit Tensor() {}

  explicit Tensor(const ArrayView<value_type, _Rank>& arr)
    : dimensions_(arr.dimensions_),
      strides_(arr.Strides()),
      size_(arr.Size()),
      data_(arr.Data())
  {}

  explicit Tensor(ArrayView<value_type, _Rank>&& arr)
    : dimensions_(std::move(arr.Dimensions())),
      strides_(std::move(arr.Strides())),
      size_(arr.Size()),
      data_(std::move(arr.Data()))
  {}


  /// View returns a "view" of the tensor, which can be a "sub-tensor" or add "broadcastable" axes.
  /// It requires that the underlying tensor's lifetime is ...
  template <size_t _ViewRank>
  auto View(const ssize_t(& axes)[_ViewRank], const ssize_t(& offsets)[_Rank]) const &
  {
    return TensorView(*this, axes, offsets);
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, _Rank>& Dimensions() const     { return dimensions_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size()                                           { return size_; }

  /// Data returns a pointer to the data buffer.
  char* Data() const                                      { return data_; }

 private:
  std::array<size_t, _Rank>   dimensions_;
  std::array<ssize_t, _Rank>  strides_;
  size_t                      size_;
  pointer                     data_;
};


/// TensorView<Tensor, Rank> implements a view of a tensor.
///
/// Note that a view cannot be created from a temporary rval; it will return a tensor.
template <PrimitiveTensor _Tensor, size_t _Rank>
class TensorView
{
 public:
  using value_type = typename _Tensor::value_type;
  using pointer = typename _Tensor::pointer;
  using const_pointer = typename _Tensor::const_pointer;
  constexpr static size_t rank = _Rank;

 private:
  // pointer_cast provides a (value) const-qualified pointer cast
  template <typename T, typename S> requires (std::is_const_v<std::remove_pointer_t<S>>)
  const std::remove_pointer_t<T>* pointer_cast(S pointer)
  {
    return reinterpret_cast<const std::remove_pointer_t<T>*>(pointer);
  }
  template <typename T, typename S> requires (!std::is_const_v<std::remove_pointer_t<S>>)
  std::remove_pointer_t<T>* pointer_cast(S pointer)
  {
    return reinterpret_cast<std::remove_pointer_t<T>*>(pointer);
  }

 public:
  // TensorView must be in the same scope and lifetime as the underlying tensor.
  TensorView() = delete;

  /// Constructor
  template <size_t _TensorRank>
  explicit TensorView(_Tensor& tensor, const ssize_t(& axes)[_Rank], const ssize_t(& offsets)[_TensorRank])
    : size_(0UL)
  {
    std::bitset<_TensorRank> handled = false;
    auto strides = tensor.Strides();
    auto dimensions    = tensor.Dimensions();

    for (ssize_t i = static_cast<ssize_t>(_Rank) - 1; i >= 0; i--)
    {
      if (axes[i] >= 0 && axes[i] < static_cast<ssize_t>(_TensorRank))
      {
        if (handled[axes[i]])
          throw std::runtime_error("axis can only be used once");

        handled.set(axes[i]);
        dimensions_[i] = dimensions[axes[i]];
        strides_[i] = strides[axes[i]];
        size_ += dimensions_[i] * strides_[i];
      }
      else if (axes[i] == Broadcast)
      {
        dimensions_[i] = 1;
        strides_[i] = 0;
      }
      else
        throw std::runtime_error("Invalid axis");
    }

    size_t offset = 0UL;
    for (size_t i = 0; i < _TensorRank; i++)
    {
      if (offsets[i] > static_cast<ssize_t>(tensor.dimensions_[i] * tensor.strides_[i]))
        throw std::runtime_error("Offset exceeds dimension");
      offset += offsets[i] * tensor.strides_[i];
    }

    data_ = reinterpret_cast<pointer>(pointer_cast<char*>(tensor.Data()) + offset);
  }


  /// operator=(Tensor) copies data from the rhs tensor (or view) into the view of the dependent tensor.
  template <AnyTensor _FromTensor> requires (_FromTensor::rank == _Rank)
  auto operator=(const _FromTensor& rhs)
  {
    copy<value_type, _Rank>(data_, rhs.Data(), dimensions_, strides_, rhs.Strides());
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _Rank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, _Rank>& Dimensions() const     { return dimensions_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, _Rank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return &data_; }
  const_pointer Data() const                              { return data_; }

 private:
  std::array<size_t, _Rank>   dimensions_;
  std::array<ssize_t, _Rank>  strides_;
  size_t                      size_;
  pointer                     data_;
};

//
// CTAD rules
//

// Rank-0

// Tensor{T} -> Rank-0 tensor with a static/local array
template <typename _Tp>
explicit Tensor(_Tp) -> Tensor<_Tp, 0>;

// Tensor{Uninitailzied<T>} -> Rank-0 tensor with a static/local array
template <typename _Tp>
explicit Tensor(Uninitialized<_Tp>) -> Tensor<_Tp, 0>;

// Tensor with Static Allocator - Brace-initializer List

// Tensor{Ts...} -> Rank-1 tensor with a static/local array (brace-initializer).
template <typename _Tp, typename... _Ts>
explicit Tensor(_Tp, _Ts...) -> Tensor<std::common_type_t<_Tp, _Ts...>, 1, StaticAllocator<sizeof...(_Ts)+1>>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <typename _Tp, size_t... _N>
Tensor(_Tp(&&... l)[_N]) -> Tensor<_Tp, 2, StaticAllocator<sizeof...(_N), std::max({_N...})>>;

// Tensor{{{...},...},...} -> Rank-3 tensor with a static/local array (brace-initializer).
template <typename _Tp, size_t... _M, size_t... _N>
explicit Tensor(_Tp(&&... l)[_M][_N]) -> Tensor<_Tp, 3, StaticAllocator<sizeof...(_M), std::max({_M...}), std::max({_N...})>>;

// Tensor with Dynamic Allocator - Paremter List

// Tensor(uint,T) -> Rank-1 tensor with a dynamically allocated buffer.
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

// Tensor(&&[], &&[]) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
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

// Tensor with Dynamic Allocator - TensorView Argument

template <typename _Tensor, size_t _Rank>
Tensor(TensorView<_Tensor, _Rank>&&) -> Tensor<typename _Tensor::value_type, _Rank>;
template <typename _Tensor, size_t _Rank>
Tensor(const TensorView<_Tensor, _Rank>&) -> Tensor<typename _Tensor::value_type, _Rank>;

// Tensor with Dynamic Allocator - Operator Argument
// TensorOp -> Tensor (move)
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, typename...> typename _TensorRT, typename _Tp, size_t _Rank, typename... _Tensors>
Tensor(_TensorOp<_TensorRT, _Tp, _Rank, _Tensors...>&&) -> Tensor<_Tp, _Rank>;

// TensorOp -> Tensor (copy)
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, typename...> typename _TensorRT, typename _Tp, size_t _Rank, typename... _Tensors>
Tensor(const _TensorOp<_TensorRT,_Tp,  _Rank, _Tensors...>&) -> Tensor<_Tp, _Rank>;

// Tensor with "NoAllocator" - ArrayView

// Tensor<ArrayView> -> Rank-N tensor for an externally managed buffer
template <typename _Tp, size_t _N>
explicit Tensor(const ArrayView<_Tp, _N>&) -> Tensor<_Tp, _N, NoAllocator>;
template <typename _Tp, size_t _N>
explicit Tensor(ArrayView<_Tp, _N>&&) -> Tensor<_Tp, _N, NoAllocator>;


} // end of namespace grid

#endif  // GRID_TENSOR_BASE_TENSOR_H
