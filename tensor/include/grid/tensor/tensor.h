//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_H
#define GRID_TENSOR_TENSOR_H

#include <functional>
#include <iterator>
#include <iostream>
#include <numeric>

#include "concepts.h"
#include "iterator.h"
#include "tensor_parameters.h"
#include "tensor_view.h"

namespace grid {

/// StaticAllocator is a special "allocator" for constant static data.
template <size_t...> struct StaticAllocator {};

/// NoAllocator is a spcial "allocator" for an externally managed buffer.
struct NoAllocator {};


// TODO: make the entire code templated for devices
struct BaseCPU {};


// The arithmetic declaraions must be specialized for different tensor types, which supports
// specializations for accelerators.

template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorAdd;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorMatMul;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorElemMul;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorRmsNorm;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorSoftMax;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorSilu;

//
// Tensor
//

template <typename T> class Array;

/// Tensor implements an "AI Tensor" that follows more typical AI implementations rather than
/// mathematical or physical definition.
///
/// @tparam T           Integral type
/// @tparam TRank       Rank of the tensor with 0: scalar, 1: vector, 2: matrix, etc.
///
/// Tensors define these member types and constexpr variables:
///
///   rank              TRank
///   value_type        T
///   allocator_type    TAllocator
///   pointer           Pointer type; depends on the implementation
///   const_pointer     Constant pointer type; depends on the implementation
///
/// Tensors also provide the following member methods:
///
///   constexpr size_t           Rank()
///   std::array<size_t, Rank>   Dimensions() const
///   std::array<ssize_t, Rank>  Strides() const
///   pointer                    Data()
///   const_pointer              Data() const
template <typename T, size_t TRank, typename... TAllocator> class Tensor;

/// Tensor<T,TRank> provides an non-optimized base implementation of tensors using the standard
/// std:allocator (new/delete).
/// Note that the implementation implicitly requires that the buffer and strides are aligned to the value type.
template <typename TDevice> struct FillFunc;
template <typename TDevice> struct CopyFunc;
template <typename T, size_t TRank>
class Tensor<T, TRank> : public Array<T>
{
  template <PrimitiveTensor P, size_t R> friend class TensorView;

  template <typename TDevice> static constexpr FillFunc<TDevice> Fill;
  template <typename TDevice> static constexpr CopyFunc<TDevice> Copy;

 public:
  using value_type = T;
  using allocator_type = std::allocator<T>;
  using pointer = value_type*;
  using reference = value_type&;
  using const_pointer = const value_type*;
  using const_reference = const value_type&;
  using array_type = Array<value_type>;
  constexpr static size_t rank = TRank;


  auto begin() const                  { return details::Iterator(this, array_type::data_); }
  auto end() const                    { return details::Iterator(this, array_type::data_, Dimensions()); }


 public:
  Tensor() = default;

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  explicit Tensor(size_t dimension, value_type init)
    : Array<value_type>(dimension * sizeof(value_type)),
      dimensions_{dimension},
      strides_{make_strides<value_type>(dimensions_)}
  {
    Fill<BaseCPU>(*this, init);
  }

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dimension, Uninitialized<value_type>)
    : Array<value_type>(dimension * sizeof(value_type)),
      dimensions_{dimension},
      strides_{make_strides<value_type>(dimensions_)}
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit Tensor(size_t dim_m, int dim_n, value_type init)
    : Array<value_type>(dim_m * dim_n * sizeof(value_type)),
      dimensions_{dim_m, dim_n},
      strides_{make_strides<value_type>(dimensions_)}
  {
    Fill<BaseCPU>(*this, init);
  }

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dim_m, int dim_n, Uninitialized<value_type>)
    : Array<value_type>(dim_m * dim_n * sizeof(value_type)),
      dimensions_{dim_m, dim_n},
      strides_{make_strides<value_type>(dimensions_)}
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(std::initializer_list<size_t>&& dimensions, value_type init)
    : Array<value_type>(std::accumulate(
          std::begin(dimensions), std::end(dimensions), sizeof(value_type), std::multiplies<size_t>())),
      dimensions_(get_array<size_t, TRank>(std::move(dimensions))),
      strides_{make_strides<value_type>(dimensions_)}
  {
    Fill<BaseCPU>(*this, init);
  }


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(std::initializer_list<size_t>&& dimensions, Uninitialized<value_type>)
    : Array<value_type>(std::accumulate(
          std::begin(dimensions), std::end(dimensions), sizeof(value_type), std::multiplies<size_t>())),
      dimensions_(get_array<size_t, TRank>(std::move(dimensions))),
      strides_{make_strides<value_type>(dimensions_)}
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with strides.
  explicit Tensor(std::initializer_list<size_t>&& dimensions,
                  std::initializer_list<ssize_t>&& strides,
                  value_type init)
    : Array<value_type>(get_buffer_size(dimensions, strides)),
      dimensions_(get_array<size_t, TRank>(std::move(dimensions))),
      strides_(get_array<ssize_t, TRank>(std::move(strides)))
  {
    Fill<BaseCPU>(*this, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with strides.
  explicit Tensor(std::initializer_list<size_t>&& dimensions,
                  std::initializer_list<ssize_t>&& strides,
                  Uninitialized<value_type>)
    : Array<value_type>(get_buffer_size(dimensions, strides)),
      dimensions_(get_array<size_t, TRank>(std::move(dimensions))),
      strides_(get_array<ssize_t, TRank>(std::move(strides)))
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(const size_t(&dimensions)[TRank], const ssize_t(&strides)[TRank], value_type init)
    : Array<value_type>(get_buffer_size(dimensions, strides)),
      dimensions_(get_array<size_t, TRank>(dimensions)),
      strides_(get_array<ssize_t, TRank>(strides))
  {
    Fill<BaseCPU>(*this, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  explicit Tensor(const size_t(&dimensions)[TRank], const ssize_t(&strides)[TRank], Uninitialized<value_type>)
    : Array<value_type>(get_buffer_size(dimensions, strides)),
      dimensions_(get_array<size_t, TRank>(dimensions)),
      strides_(get_array<ssize_t, TRank>(strides))
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer.
  explicit Tensor(std::array<size_t, TRank> dimensions, value_type init)
    : Array<value_type>(std::accumulate(
          begin(dimensions), end(dimensions), sizeof(value_type), std::multiplies<size_t>())),
      dimensions_(dimensions),
      strides_(make_strides<value_type>(dimensions))
  {
    Fill<BaseCPU>(*this, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with padding.
  explicit Tensor(std::array<size_t, TRank> dimensions,
                  std::array<ssize_t, TRank> strides,
                  value_type init)
    : Array<value_type>(get_buffer_size(dimensions, strides)),
      dimensions_{dimensions},
      strides_{strides}
  {
    Fill<BaseCPU>(*this, init);
  }

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer.
  /// Note: assumes strides are type-aligned.
  explicit Tensor(std::array<size_t, TRank> dimensions, Uninitialized<value_type>)
    : Array<value_type>(std::accumulate(
          std::begin(dimensions), std::end(dimensions), sizeof(value_type), std::multiplies<size_t>())),
      dimensions_{dimensions},
      strides_{make_strides<value_type>(dimensions)}
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with padding.
  explicit Tensor(std::array<size_t, TRank> dimensions,
                  std::array<ssize_t, TRank> strides,
                  Uninitialized<value_type>)
    : Array<value_type>(get_buffer_size(dimensions, strides)),
      dimensions_{dimensions},
      strides_{strides}
  {}

  /// Copy constructor
  // TODO: "flatten" new array?
  Tensor(const Tensor& other)
    : Array<value_type>(other.size_),
      dimensions_{other.Dimensions()},
      strides_{other.Strides()}
  {
    Copy<BaseCPU>(other, this->begin());
  }

  /// Move constructor
  Tensor(Tensor&& other)
    : Array<value_type>(std::move(static_cast<Array<value_type>&&>(other))),
      dimensions_{std::move(other.dimensions_)},
      strides_{std::move(other.strides_)}
  {}

  // Constructors for converting from a tensor operator.
  template <AnyOperator TOperator>
  Tensor(TOperator&& functor) : Tensor{std::move(functor())} {};

  template <AnyOperator TOperator>
  Tensor(const TOperator& functor) : Tensor{functor()} {};


  /// Assign operator
  template <PrimitiveTensor TTensor>
  Tensor& operator=(const TTensor& other)
  {
    array_type::Realloc(other.Size());

    dimensions_ = other.Dimensions();
    strides_ = other.Strides();
    Copy<BaseCPU>(other, this->begin());

    return *this;
  }

  /// Move-assign is only supported from the same type
  Tensor& operator=(Tensor&& other)
  {
    dimensions_ = other.Dimensions();
    strides_ = other.Strides();
    Array<value_type>::operator=(std::move(other));
    return *this;
  }

  /// Operator assign
  template <AnyOperator TOperator>
  Tensor& operator=(TOperator&& oper)
  {
    return operator=(std::forward<TOperator>(oper)());
  }

  template <AnyOperator TOperator>
  Tensor& operator+=(TOperator&& oper)
  {
    return operator=(TensorAdd(*this, std::forward<TOperator>(oper)()));
  }


  /// View returns a view of the proivded tensor.
  template <typename... Ts>
  auto View(Ts&&... slices)
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  template <typename... Ts>
  auto View(Ts&&... slices) const
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides)
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }

  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides) const
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return TRank; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, TRank>& Dimensions() const     { return dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, TRank>& Strides() const       { return strides_; }

 private:
  std::array<size_t, TRank>         dimensions_;
  std::array<ssize_t, TRank>        strides_;
};

/// Tensor<T, 0, 1> is a specialization of a rank-0 tensor.
template <typename T>
class Tensor<T, 0>
{
 public:
  using value_type = T;
  using allocator_type = StaticAllocator<1>;
  using pointer = value_type*;
  using reference = value_type&;
  using const_pointer = const value_type*;
  using const_reference = const value_type&;
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
template <typename T, size_t N>
class Tensor<T, 1, StaticAllocator<N>>
{
 public:
  using value_type = T;
  using allocator_type = StaticAllocator<N>;
  using pointer = const value_type*;
  using reference = const value_type&;
  using const_pointer = const value_type*;
  using const_reference = const value_type&;
  constexpr static size_t rank = 1UL;

  /// Constructor for a rank-1 tensor (vector) with brace initialization.
  explicit Tensor(std::initializer_list<value_type>&& init)
    : dimensions_{N},
      strides_{sizeof(value_type)},
      array_(get_array<value_type, N>(std::move(init)))
  {}


  /// View returns a view of the proivded tensor.
  template <typename... Ts>
  auto View(Ts&&... slices)
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  template <typename... Ts>
  auto View(Ts&&... slices) const
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides)
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }

  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides) const
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
  size_t Size() const                                     { return sizeof(value_type) * N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }

 private:
  std::array<size_t, 1>     dimensions_;
  std::array<ssize_t, 1>    strides_;
  std::array<value_type, N> array_;
};


/// Tensor<T, M, N> is a specialization of a rank-2 tensor (matrix) for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename T, size_t M, size_t N>
class Tensor<T, 2, StaticAllocator<M, N>>
{
 public:
  using value_type = T;
  using pointer = const value_type*;
  using reference = const value_type&;
  using const_pointer = const value_type*;
  using const_reference = const value_type&;
  constexpr static size_t rank = 2UL;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit Tensor(std::initializer_list<std::initializer_list<value_type>>&& init)
    : dimensions_{M, N},
      strides_{ M != 1 ? sizeof(value_type) * N : 0, sizeof(value_type)},
      array_(get_array<value_type, M, N>(std::move(init)))
  {}


  /// View returns a view of the proivded tensor.
  template <typename... Ts>
  auto View(Ts&&... slices)
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  template <typename... Ts>
  auto View(Ts&&... slices) const
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides)
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }

  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides) const
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
  size_t Size() const                                     { return sizeof(value_type) * M * N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }

 private:
  std::array<size_t, 2>           dimensions_;
  std::array<ssize_t, 2>          strides_;
  std::array<value_type, M * N> array_;
};


/// Tensor<T, C, M, N> is a specialization of a rank-3 tensor for a 'static' array.
/// Note that the brace-initializer form of Tensors don't support padding.
template <typename T, size_t C, size_t M, size_t N>
class Tensor<T, 3, StaticAllocator<C, M, N>>
{
 public:
  using value_type = T;
  using allocator_type = StaticAllocator<C, M, N>;
  using pointer = const value_type*;
  using reference = const value_type&;
  using const_pointer = const value_type*;
  using const_reference = const value_type&;
  constexpr static size_t rank = 3UL;

  /// Constructor for a rank-2 (matrix) brace initialization.
  explicit Tensor(std::initializer_list<std::initializer_list<std::initializer_list<value_type>>>&& init)
    : dimensions_{C, M, N},
      strides_{ C != 1 ? sizeof(value_type) * M * N : 0,
                M != 1 ? sizeof(value_type) * N : 0,
                N != 1 ? sizeof(value_type) : 0},
      array_(get_array<value_type, C, M, N>(std::move(init)))
  {}


  /// View returns a view of the proivded tensor.
  template <typename... Ts>
  auto View(Ts&&... slices)
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  template <typename... Ts>
  auto View(Ts&&... slices) const
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides)
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }

  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides) const
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
  size_t Size() const                                     { return sizeof(value_type) * C * M * N; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }


  std::array<size_t, 3>                 dimensions_;
  std::array<ssize_t, 3>                strides_;
  std::array<value_type, C * M * N>  array_;
};


/// Tensor<T, Rank, NoAllocator> is a tensor for an externally managed buffer
template <typename T, size_t TRank>
class Tensor<T, TRank, NoAllocator>
{
 public:
  using value_type = T;
  using allocator_type = NoAllocator;
  using pointer = const value_type*;
  using reference = const value_type&;
  using const_pointer = const value_type*;
  using const_reference = const value_type&;
  constexpr static size_t rank = TRank;

  explicit Tensor() {}

  explicit Tensor(const size_t(&& dimensions)[TRank], const std::tuple<pointer, size_t>& array)
    : dimensions_(std::to_array(dimensions)),
      strides_{make_strides<value_type>(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(std::get<0>(array))
  {
    if (size_ > std::get<1>(array))
      throw std::runtime_error("dimensions exceed allotted size: " + std::to_string(size_) + " > " +
          std::to_string(std::get<1>(array)));
  }

  explicit Tensor(const std::array<size_t, TRank>& dimensions, const std::tuple<pointer, size_t>& array)
    : dimensions_(dimensions),
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


  /// View returns a view of the proivded tensor.
  template <typename... Ts>
  auto View(Ts&&... slices)
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  template <typename... Ts>
  auto View(Ts&&... slices) const
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides)
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }

  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides) const
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }



  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return TRank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, TRank>& Dimensions() const     { return dimensions_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, TRank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size()                                           { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return data_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return data_; }

 private:
  std::array<size_t, TRank>   dimensions_;
  std::array<ssize_t, TRank>  strides_;
  size_t                      size_;
  pointer                     data_;
};


//
// CTAD rules
//

// Rank-0

// Tensor{T} -> Rank-0 tensor with a static/local array
template <Arithmetic T>
explicit Tensor(T) -> Tensor<T, 0>;

// Tensor{Uninitailzied<T>} -> Rank-0 tensor with a static/local array
template <Arithmetic T>
explicit Tensor(Uninitialized<T>) -> Tensor<T, 0>;

// Tensor with Static Allocator - Brace-initializer List

// Tensor{Ts...} -> Rank-1 tensor with a static/local array (brace-initializer).
template <Arithmetic T, typename... Ts>
explicit Tensor(T, Ts...) -> Tensor<std::common_type_t<T, Ts...>, 1, StaticAllocator<sizeof...(Ts)+1>>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <Arithmetic T, size_t... N>
Tensor(T(&&... l)[N]) -> Tensor<T, 2, StaticAllocator<sizeof...(N), std::max({N...})>>;

// Tensor{{{...},...},...} -> Rank-3 tensor with a static/local array (brace-initializer).
template <Arithmetic T, size_t... M, size_t... N>
explicit Tensor(T(&&... l)[M][N]) -> Tensor<T, 3, StaticAllocator<sizeof...(M), std::max({M...}), std::max({N...})>>;

// Tensor with Dynamic Allocator - Paremter List

// Tensor(uint,T) -> Rank-1 tensor with a dynamically allocated buffer.
template <Arithmetic T>
explicit Tensor(size_t, T) -> Tensor<T, 1>;

// Tensor(uint, Uninitialized<T>) -> Rank-1 tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T>
explicit Tensor(size_t, Uninitialized<T>) -> Tensor<T, 1>;

// Tensor(uint, uint, T) -> Rank-2 tensor with a dynamically allocated buffer.
template <Arithmetic T>
explicit Tensor(size_t, size_t, T) -> Tensor<T, 2>;

// Tensor(uint, Uninitialized<T>) -> Rank-2 tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T>
explicit Tensor(size_t, size_t, Uninitialized<T>) -> Tensor<T, 2>;

// Tensor(&[], &[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N>
explicit Tensor(const size_t(&)[N], const ssize_t(&)[N], T) -> Tensor<T, N>;

// Tensor(&[], &[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N>
explicit Tensor(const size_t(&)[N], const ssize_t(&)[N], Uninitialized<T>) -> Tensor<T, N>;

// Tensor(&&[], &&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N>
explicit Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N>;

// Tensor(&&[], &&[]) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N>
explicit Tensor(size_t(&&)[N], ssize_t(&&)[N], Uninitialized<T>) -> Tensor<T, N>;

// Tensor(&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N>
explicit Tensor(const size_t(&)[N], T) -> Tensor<T, N>;

// Tensor(&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N>
explicit Tensor(const size_t(&)[N], Uninitialized<T>) -> Tensor<T, N>;

// Tensor(&&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N>
explicit Tensor(const size_t(&&)[N], T) -> Tensor<T, N>;

// Tensor(&&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N>
explicit Tensor(const size_t(&&)[N], Uninitialized<T>) -> Tensor<T, N>;

// Tensor(array, T)
template <Arithmetic T, size_t N>
Tensor(std::array<size_t, N>, T) -> Tensor<T, N>;

// Tensor(array, array, T)
template <Arithmetic T, size_t N>
explicit Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<T, N>;

// Tensor(array, Uninitialized<T>)
template <Arithmetic T, size_t N>
explicit Tensor(std::array<size_t, N>, Uninitialized<T>) -> Tensor<T, N>;

// Tensor(array, array, Uninitialized<T>)
template <Arithmetic T, size_t N>
explicit Tensor(std::array<size_t, N>, std::array<ssize_t, N>, Uninitialized<T>) -> Tensor<T, N>;

// Tensor with Dynamic Allocator - TensorView Argument

template <typename TTensor, size_t TRank>
Tensor(TensorView<TTensor, TRank>&&) -> Tensor<typename TTensor::value_type, TRank>;
template <typename TTensor, size_t TRank>
Tensor(const TensorView<TTensor, TRank>&) -> Tensor<typename TTensor::value_type, TRank>;

// Tensor with Dynamic Allocator - Operator Argument
// Tensor(Operator) -> Tensor (move)
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename TOperator,
          template <typename, size_t, typename...> typename TTensor, typename T, size_t TRank, typename... TTensors>
Tensor(TOperator<TTensor, T, TRank, TTensors...>&&) -> Tensor<T, TRank>;

// Tensor(Operator) -> Tensor (copy)
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename TOperator,
          template <typename, size_t, typename...> typename TTensor, typename T, size_t TRank, typename... TTensors>
Tensor(const TOperator<TTensor, T,  TRank, TTensors...>&) -> Tensor<T, TRank>;

// Tensor with "NoAllocator"
template <Arithmetic T, size_t N>
explicit Tensor(const size_t(&)[N], const std::tuple<T*, size_t>&) -> Tensor<T, N, NoAllocator>;
template <Arithmetic T, size_t N>
explicit Tensor(const std::array<size_t, N>&, const std::tuple<T*, size_t>&) -> Tensor<T, N, grid::NoAllocator>;


//
// Tensor basic arithmetic operations
//

//
// Arithmentic operator overloading
//

// operator+ (TensorType, TensorType)
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto operator+(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return TensorAdd(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

// operator* (TensorType, TensorType) -> ElemMul, requires same rank
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto operator*(TTensor1&& tensor1, TTensor2&& tensor2)
requires (std::decay_t<TTensor1>::rank == std::decay_t<TTensor2>::rank)
{
  return TensorElemMul(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

// operator* (TensorType, TensorType) -> Scale, if one Tensor has rank0
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto operator*(TTensor1&& tensor1, TTensor2&& tensor2)
requires (std::decay_t<TTensor1>::rank == 0 || std::decay_t<TTensor2>::rank == 0)
{
  return TensorMatMul(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

// operator* (TensorType, arithmetic)
template <TensorConvertible TTensor, Arithmetic T>
auto operator*(TTensor&& tensor, T scalar)
{
  return TensorMatMul(std::forward<TTensor>(tensor), scalar);
}

// operator* (arithmetic, TensorType)
template <Arithmetic T, TensorConvertible TTensor>
auto operator*(T scalar, TTensor&& tensor)
{
  return TensorMatMul(scalar, std::forward<TTensor>(tensor));
}

} // end of namespace grid

/// operator<<(TENSOR) overloads the output operator for tensors.
std::ostream& operator<<(std::ostream& os, const grid::AnyTensor auto& tensor)
{
  using value_type = typename std::remove_reference_t<decltype(tensor)>::value_type;
  size_t rank = tensor.Rank();

  auto dimensions = tensor.Dimensions();
  auto strides = tensor.Strides();

  std::function<void(int, const value_type*)> print;
  print = [&os, &dimensions, &strides, &print, &rank](size_t index, const value_type* ptr) {
    os << "{ ";
    if (index < rank - 1)
    {
      for (size_t i = dimensions[index]; i > 0; i--)
      {
        print(index + 1, ptr);
        if (i != 1)
          os << ", ";
        else
          os << " }";
        reinterpret_cast<const char*&>(ptr) += strides[index];
      }
    }
    else
    {
      for (size_t i = dimensions[rank-1]; i > 0; i--)
      {
        os << *ptr;
        if (i != 1)
          os << ", ";
        else
          os << " }";
        reinterpret_cast<const char*&>(ptr) += strides[rank-1];
      }
    }
  };
  const value_type* ptr = reinterpret_cast<const value_type*>(tensor.Data());
  if (rank > 0)
    print(0, ptr);
  else
    os << "{ " << *ptr << " }";

  os << std::flush;

  return os;
}

#endif  // GRID_TENSOR_TENSOR_H
