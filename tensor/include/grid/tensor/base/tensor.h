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
#include "tensor_view.h"

#include "copy.h"

namespace grid {

namespace {

template <typename _Tp>
inline void initialize(_Tp* dst, std::span<size_t, 1> dimensions, std::span<ssize_t, 1> strides, _Tp init)
{
  for (size_t i = 0; i < dimensions[0]; i++, reinterpret_cast<char*&>(dst) += strides[0])
    *dst = init;
}

template <typename _Tp, size_t _N>
inline void initialize(_Tp* dst, std::span<size_t, _N> dimensions, std::span<ssize_t, _N> strides, _Tp init)
{
  for (size_t i = 0; i < dimensions[0]; i++, reinterpret_cast<char*&>(dst) += strides[0])
    initialize(dst,
        std::span<size_t, _N - 1>(dimensions.begin() + 1, dimensions.end()),
        std::span<ssize_t, _N - 1>(strides.begin() + 1, strides.end()),
        init);
}

} // end of namespace details


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

 public:
  /// Default constructor
  Tensor() = default;

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  explicit Tensor(size_t dim, value_type init)
    : dimensions_{dim},
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Tensor rank1: %p\n", data_);
    initialize(data_, std::span{dimensions_}, std::span{strides_}, init);
  }

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dim, Uninitialized<value_type>)
    : dimensions_{dim},
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Tensor rank1 uninit: %p\n", data_);
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit Tensor(size_t dim_m, int dim_n, value_type init)
    : dimensions_{dim_m, dim_n},
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    initialize(data_, std::span{dimensions_}, std::span{strides_}, init);
    //printf("Tensor rank2: %p\n", data_);
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dim_m, int dim_n, Uninitialized<value_type>)
    : dimensions_{dim_m, dim_n},
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Tensor rank2 uninit: %p\n", data_);
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(std::initializer_list<size_t>&& dimensions, value_type init)
    : dimensions_(get_array<size_t, _Rank>(std::move(dimensions))),
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Tensor rank2: %p\n", data_);
    initialize(data_, std::span{dimensions_}, std::span{strides_}, init);
  }


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(std::initializer_list<size_t>&& dimensions, Uninitialized<value_type>)
    : dimensions_(get_array<size_t, _Rank>(std::move(dimensions))),
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Tensor rank2 uninit: %p\n", data_);
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with strides.
  explicit Tensor(std::initializer_list<size_t>&& dimensions,
                  std::initializer_list<ssize_t>&& strides,
                  value_type init)
    : dimensions_(get_array<size_t, _Rank>(std::move(dimensions))),
      strides_(get_array<ssize_t, _Rank>(std::move(strides))),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Tensor rankN: %p\n", data_);
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
  {
    //printf("Tensor rankN uninit: %p\n", data_);
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(const size_t(&dimensions)[_Rank], const ssize_t(&stride)[_Rank], value_type init)
    : dimensions_(get_array<size_t, _Rank>(dimensions)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Tensor rankN(b): %p\n", data_);
    initialize(data_, std::span{dimensions_}, std::span{strides_}, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  explicit Tensor(const size_t(&dimensions)[_Rank], const ssize_t(&stride)[_Rank], Uninitialized<value_type>)
    : dimensions_(get_array<size_t, _Rank>(dimensions)),
      strides_(get_array<ssize_t, _Rank>(stride)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Tensor rankN(b) uninit: %p\n", data_);
  }


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer.
  explicit Tensor(std::array<size_t, _Rank> dimensions, value_type init)
    : dimensions_(dimensions),
      strides_(make_strides<value_type>(dimensions)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Tensor rankN(c): %p\n", data_);
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
    //printf("Tensor rankN(d): %p\n", data_);
    initialize<_Rank>(data_, dimensions_, strides_, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer.
  /// Note: assumes strides are type-aligned.
  explicit Tensor(std::array<size_t, _Rank> dimensions, Uninitialized<value_type>)
    : dimensions_{dimensions},
      strides_{make_strides<value_type>(dimensions)},
      size_(dimensions_[0] * strides_[0]),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Tensor rankN(d) uninit: %p\n", data_);
  }


  /// Constructor from a 'trivially copyable' tensor.
  /// Note: assumes strides are type-aligned.
  template <PrimitiveTensor _Tensor>
  Tensor(const _Tensor& other)
    : dimensions_(other.Dimensions()),
      strides_(other.Strides()),
      size_(other.Size()),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Copy constructor %p %p\n", data_, other.Data());
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
  {
    //printf("Tensor rankN(e) uninit: %p\n", data_);
  }

  // Constructors for converting from a tensor operator.
  template <AnyOperator Operator>
  //Tensor(Operator&& functor) : Tensor{std::move(functor())} {};
  Tensor(Operator&& functor) : Tensor{functor()} {};

  template <AnyOperator Operator>
  Tensor(const Operator& functor) : Tensor{functor()} {};


  /// Copy constructor
  Tensor(const Tensor& other)
    : dimensions_{other.Dimensions()},
      strides_{other.Strides()},
      size_(other.Size()),
      data_(new value_type[size_ / sizeof(value_type)])
  {
    //printf("Copy %p %p\n", data_, other.Data());
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
    //std::cout << "~Tensor: " << data_ << std::endl;
    if (data_ != nullptr)
      delete[] data_;
  }


  /// Assign operator
  template <PrimitiveTensor _Tensor>
  Tensor& operator=(const _Tensor& other)
  {
    // FIXME why delete local buffer?
    //printf("Copy %p %lu %lu\n", data_, size_, other.Size());
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
    data_ = other.data_;
    other.data_ = nullptr;

    return *this;
  }

  /// Operator assign
  template <AnyOperator _Operator>
  Tensor& operator=(_Operator&& oper)
  {
    return operator=(std::forward<_Operator>(oper)());
  }

  template <AnyOperator _Operator>
  Tensor& operator+=(_Operator&& oper)
  {
    return operator=(TensorAdd(*this, std::forward<_Operator>(oper)()));
  }


  /// Span returns a view to a subset of the tensor retaining the rank of the tensor.
  auto Span(const size_t(& dimensions)[_Rank], const size_t(& offsets)[_Rank])
  {
    return view:: Span(*this, std::to_array(dimensions), std::to_array(offsets));
  }

  auto Span(const size_t(& dimensions)[_Rank], const size_t(& offsets)[_Rank]) const
  {
    return view::Span(*this, std::to_array(dimensions), std::to_array(offsets));
  }

  /// Rearrange returns a view of a tensor where the axes have been rearrange, dropped, or
  /// expande (broadcast).
  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank],
                 const size_t(& dimensions)[_Rank],
                 const size_t(& offsets)[_Rank])
  {
    return view::Rearrange(*this, axes, std::to_array(dimensions), std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank],
                 const size_t(& dimensions)[_Rank],
                 const size_t(& offsets)[_Rank]) const
  {
    return view::Rearrange(*this, axes, std::to_array(dimensions), std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank], const size_t(& offsets)[_Rank] = {0})
  {
    return view::Rearrange(*this, std::to_array(axes), {0}, std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank], const size_t(& offsets)[_Rank] = {0}) const
  {
    return view::Rearrange(*this, std::to_array(axes), {0}, std::to_array(offsets));
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t _ViewRank>
  auto Reshape(const std::array<size_t, _ViewRank>& dimensions,
               const std::array<ssize_t, _ViewRank>& strides)
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }

  template <size_t _ViewRank>
  auto Reshape(const std::array<size_t, _ViewRank>& dimensions,
               const std::array<ssize_t, _ViewRank>& strides) const
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
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

  /// Data returns a pointer to the data buffer.
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
  void some_func() {}
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


  /// Span returns a view to a subset of the tensor retaining the rank of the tensor.
  auto Span(const size_t(& dimensions)[1], const size_t(& offsets)[1])
  {
    return view::Span(*this, std::to_array(dimensions), std::to_array(offsets));
  }

  auto Span(const size_t(& dimensions)[1], const size_t(& offsets)[1]) const
  {
    return view::Span(*this, std::to_array(dimensions), std::to_array(offsets));
  }

  /// Rearrange returns a view of a tensor where the axes have been rearrange, dropped, or
  /// expande (broadcast).
  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank],
                 const size_t(& dimensions)[1],
                 const size_t(& offsets)[1])
  {
    return view::Rearrange(*this, std::to_array(axes), std::to_array(dimensions), std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank],
                 const size_t(& dimensions)[1],
                 const size_t(& offsets)[1]) const
  {
    return view::Rearrange(*this, std::to_array(axes), std::to_array(dimensions), std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank], const size_t(& offsets)[1] = {0})
  {
    return view::Rearrange(*this, std::to_array(axes), {0}, std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank], const size_t(& offsets)[1] = {0}) const
  {
    return view::Rearrange(*this, std::to_array(axes), {0}, std::to_array(offsets));
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t _ViewRank>
  auto Reshape(const std::array<size_t, _ViewRank>& dimensions,
               const std::array<ssize_t, _ViewRank>& strides)
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }

  template <size_t _ViewRank>
  auto Reshape(const std::array<size_t, _ViewRank>& dimensions,
               const std::array<ssize_t, _ViewRank>& strides) const
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
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

  /// Span returns a view to a subset of the tensor retaining the rank of the tensor.
  auto Span(const size_t(& dimensions)[2UL], const size_t(& offsets)[2UL])
  {
    return view:: Span(*this, std::to_array(dimensions), std::to_array(offsets));
  }

  auto Span(const size_t(& dimensions)[2UL], const size_t(& offsets)[2UL]) const
  {
    return view:: Span(*this, std::to_array(dimensions), std::to_array(offsets));
  }

  /// Rearrange returns a view of a tensor where the axes have been rearrange, dropped, or
  /// expande (broadcast).
  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank],
                 const size_t(& dimensions)[2],
                 const size_t(& offsets)[2])
  {
    return view::Rearrange(*this, axes, std::to_array(dimensions), std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank],
                 const size_t(& dimensions)[2],
                 const size_t(& offsets)[2]) const
  {
    return view::Rearrange(*this, axes, std::to_array(dimensions), std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank], const size_t(& offsets)[2] = {0})
  {
    return view::Rearrange(*this, std::to_array(axes), {0}, std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank], const size_t(& offsets)[2] = {0}) const
  {
    return view::Rearrange(*this, std::to_array(axes), {0}, std::to_array(offsets));
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t _ViewRank>
  auto Reshape(const std::array<size_t, _ViewRank>& dimensions,
               const std::array<ssize_t, _ViewRank>& strides)
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }

  template <size_t _ViewRank>
  auto Reshape(const std::array<size_t, _ViewRank>& dimensions,
               const std::array<ssize_t, _ViewRank>& strides) const
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
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

  /// Span returns a view to a subset of the tensor retaining the rank of the tensor.
  /// @param dimensions  Dimensions for each axis
  auto Span(const size_t(& dimensions)[3UL])
  {
    return view::Span(*this, std::to_array(dimensions));
  }

  /// Span returns a view to a subset of the tensor with an offset retaining the rank of the tensor.
  /// @param dimensions  Dimensions for each axis
  /// @param offsets  Offsets of the subset for each axis
  auto Span(const size_t(& dimensions)[3UL], const size_t(& offsets)[3UL]) const
  {
    return view::Span(*this, std::to_array(dimensions), std::to_array(offsets));
  }

  /// Rearrange returns a view of a tensor where the axes have been rearrange, dropped, or
  /// expande (broadcast).
  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank],
                 const size_t(& dimensions)[3],
                 const size_t(& offsets)[3]) const
  {
    return view::Rearrange(*this, axes, std::to_array(dimensions), std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank], const size_t(& offsets)[3] = {0}) const
  {
    return view::Rearrange(*this, std::to_array(axes), {0}, std::to_array(offsets));
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t _ViewRank>
  auto Reshape(const std::array<size_t, _ViewRank>& dimensions,
               const std::array<ssize_t, _ViewRank>& strides) const
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }


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
#if 1
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
#endif
  explicit Tensor(const size_t(&& dimensions)[_Rank], std::tuple<pointer, size_t>&& array)
    : dimensions_(std::to_array(dimensions)),
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(std::get<0>(array))
  {
    if (size_ > std::get<1>(array))
      throw std::runtime_error("dimensions exceed allotted size: " + std::to_string(size_) + " > " + 
          std::to_string(std::get<1>(array)));
  }


  Tensor& operator=(Tensor&& other)
  {
    dimensions_ = other.dimensions_;
    strides_ = other.strides_;
    size_ = other.size_;
    data_ = other.data_;
    return *this;
  }

  Tensor& operator=(Tensor& other)
  {
    dimensions_ = other.dimensions_;
    strides_ = other.strides_;
    size_ = other.size_;
    data_ = other.data_;
    return *this;
  }

  Tensor(Tensor& other)
    : dimensions_(other.dimensions_),
      strides_(other.strides_),
      size_(other.size_),
      data_(other.data_)
  {
  }

  Tensor(Tensor&& other)
    : dimensions_(other.dimensions_),
      strides_(other.strides_),
      size_(other.size_),
      data_(other.data_)
  {
    other.data_ = nullptr;
  }

  /// Span returns a view to a subset of the tensor retaining the rank of the tensor.
  auto Span(const size_t(& dimensions)[_Rank], const size_t(& offsets)[_Rank])
  {
    return view:: Span(*this, std::to_array(dimensions), std::to_array(offsets));
  }

  auto Span(const size_t(& dimensions)[_Rank], const size_t(& offsets)[_Rank]) const
  {
    return view:: Span(*this, std::to_array(dimensions), std::to_array(offsets));
  }

  /// Rearrange returns a view of a tensor where the axes have been rearrange, dropped, or
  /// expande (broadcast).
  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank],
                 const size_t(& dimensions)[_Rank],
                 const size_t(& offsets)[_Rank])
  {
    return view::Rearrange(*this, axes, std::to_array(dimensions), std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank],
                 const size_t(& dimensions)[_Rank],
                 const size_t(& offsets)[_Rank]) const
  {
    return view::Rearrange(*this, axes, std::to_array(dimensions), std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank], const size_t(& offsets)[_Rank] = {0})
  {
#if 0
    return TensorView<std::remove_cvref_t<decltype(*this)>, _ViewRank>::Rearrange(
        *this, std::to_array(axes), std::to_array(indices), {0}, {0});
#endif
    return view::Rearrange(*this, std::to_array(axes), {0}, std::to_array(offsets));
  }

  template <size_t _ViewRank>
  auto Rearrange(const ssize_t(& axes)[_ViewRank], const size_t(& offsets)[_Rank] = {0}) const
  {
    return view::Rearrange(*this, std::to_array(axes), {0}, std::to_array(offsets));
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t _ViewRank>
  auto Reshape(const std::array<size_t, _ViewRank>& dimensions,
               const std::array<ssize_t, _ViewRank>& strides)
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }

  template <size_t _ViewRank>
  auto Reshape(const std::array<size_t, _ViewRank>& dimensions,
               const std::array<ssize_t, _ViewRank>& strides) const
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
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
template <typename _Tp, size_t = 1, typename... _Ts>
explicit Tensor(_Tp, _Ts...) -> Tensor<std::common_type_t<_Tp, _Ts...>, 1, StaticAllocator<sizeof...(_Ts)+1>>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <typename _Tp, size_t... _N>
Tensor(_Tp(&&... l)[_N]) -> Tensor<_Tp, 2, StaticAllocator<sizeof...(_N), std::max({_N...})>>;

// Tensor{{{...},...},...} -> Rank-3 tensor with a static/local array (brace-initializer).
template <typename _Tp, size_t... _M, size_t... _N>
explicit Tensor(_Tp(&&... l)[_M][_N]) -> Tensor<_Tp, 3, StaticAllocator<sizeof...(_M), std::max({_M...}), std::max({_N...})>>;

// Tensor with Dynamic Allocator - Paremter List

// Tensor(uint,T) -> Rank-1 tensor with a dynamically allocated buffer.
template <Scalar _Tp>
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
template <Scalar _Tp, size_t _N>
explicit Tensor(const size_t(&)[_N], _Tp) -> Tensor<_Tp, _N>;

// Tensor(&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <typename _Tp, size_t _N>
explicit Tensor(const size_t(&)[_N], Uninitialized<_Tp>) -> Tensor<_Tp, _N>;

// Tensor(&&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Scalar _Tp, size_t _N>
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

// FIXME
// Tensor with Dynamic Allocator - Operator Argument
// TensorOp -> Tensor (move)
#if 1
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, typename...> typename _TensorRT, typename _Tp, size_t _Rank, typename... _Tensors>
Tensor(_TensorOp<_TensorRT, _Tp, _Rank, _Tensors...>&&) -> Tensor<_Tp, _Rank>;

// TensorOp -> Tensor (copy)
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, typename...> typename _TensorRT, typename _Tp, size_t _Rank, typename... _Tensors>
Tensor(const _TensorOp<_TensorRT,_Tp,  _Rank, _Tensors...>&) -> Tensor<_Tp, _Rank>;
#endif
// Tensor with "NoAllocator" - ArrayView
// FIXME: memory mapped
#if 0
template <typename _Tp, size_t _N>
explicit Tensor(MMap&, size_t, 
#endif
// Tensor<ArrayView> -> Rank-N tensor for an externally managed buffer
template <typename _Tp, size_t _N>
explicit Tensor(const ArrayView<_Tp, _N>&) -> Tensor<_Tp, _N, NoAllocator>;
template <typename _Tp, size_t _N>
explicit Tensor(ArrayView<_Tp, _N>&&) -> Tensor<_Tp, _N, NoAllocator>;


template <typename _Tp, size_t _N>
explicit Tensor(const size_t(&)[_N], std::tuple<_Tp*, size_t>&) -> Tensor<_Tp, _N, NoAllocator>;
template <typename _Tp, size_t _N>
explicit Tensor(const size_t(&)[_N], std::tuple<_Tp*, size_t>&&) -> Tensor<_Tp, _N, NoAllocator>;

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_TENSOR_H
