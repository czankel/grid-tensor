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
#include <numeric>
#include <tuple>

#include "../array.h"
#include "../tensor_parameters.h"

#include "copy.h"

namespace grid {


namespace {

template <typename T>
inline void initialize(T* dst, std::span<const size_t, 1> dimensions, std::span<const ssize_t, 1> strides, const T& init)
{
  for (size_t i = 0; i < dimensions[0]; i++, reinterpret_cast<char*&>(dst) += strides[0])
    *dst = init;
}

template <typename T, size_t N>
inline void initialize(T* dst, std::span<const size_t, N> dimensions, std::span<const ssize_t, N> strides, const T& init)
{
  for (size_t i = 0; i < dimensions[0]; i++, reinterpret_cast<char*&>(dst) += strides[0])
    initialize(dst,
        std::span<const size_t, N - 1>(dimensions.begin() + 1, dimensions.end()),
        std::span<const ssize_t, N - 1>(strides.begin() + 1, strides.end()),
        init);
}

} // end of namespace details

template <typename T>
class Array
{
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;

 public:
  Array() = default;

  // Explicity disallow copy construction as Array isn't fully aware of any buffer structure.
  Array(const Array& other) = delete;

  // @brief Move constructor.
  Array(Array&& other) : size_(other.size_), data_(std::move(other.data_)) { other.data_ = nullptr; }

  // @brief Allocates a buffer of the provided size.
  Array(size_t size)
    : size_(size),
      data_(static_cast<pointer>(operator new[](size_, std::align_val_t(16))))
  {}

  ~Array()
  {
    if (data_ != nullptr)
      operator delete[](data_, std::align_val_t(16));
  }

  Array& operator=(Array&& other)
  {
    if (data_ != nullptr)
      operator delete[](data_, std::align_val_t(16));

    size_ = other.size_;
    data_ = std::move(other.data_);
    other.data_ = nullptr;

    return *this;
  }

  Array& operator=(const Array& other) = delete;


  /// Resize resizes the buffer of the Array. This will destroy
  Array& Realloc(size_t size)
  {
    if (size != size_)
    {
      if (data_ != nullptr)
        operator delete[](data_, std::align_val_t(16));
      data_ = static_cast<pointer>(operator new[](size_, std::align_val_t(16)));
    }

    return *this;
  }


  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return data_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return data_; }

 protected:
  size_t  size_;
  pointer data_;
};


// FIXME: the coordinates of the first tensor must be added to the
template <>
struct FillFunc<BaseCPU>
{
  template<typename T, std::output_iterator<const T&> O, std::sentinel_for<O> S>
  constexpr O operator()(O first, S last, const T& value) const
  {
    constexpr size_t rank = O::rank;

    // TODO, identify if first is {0} and skip loop
    auto dimensions = last.Coordinates();
    auto& subtrahend = first.Coordinates();
    for (size_t i = 0; i < rank; i++)
      dimensions[i] -= subtrahend[i];

    initialize(&*first, std::span<const size_t, rank>{dimensions}, std::span{first.Strides()}, value);
    return first;
  }

  template<typename T, std::ranges::output_range<const T&> R>
  constexpr std::ranges::borrowed_iterator_t<R> operator()(R&& r, const T& value) const
  {
    return (*this)(std::ranges::begin(r), std::ranges::end(r), value);
  }
};



template <>
struct CopyFunc<BaseCPU>
{
  template<std::input_iterator I, std::sentinel_for<I> S, std::weakly_incrementable O>
  requires std::indirectly_copyable<I, O>
  constexpr std::ranges::copy_result<I, O> operator()(I first, S last, O result) const
  {
    constexpr size_t rank = O::rank;

    // TODO, identify if first is {0} and skip loop
    auto dimensions = last.Coordinates();
    auto& subtrahend = first.Coordinates();
    for (size_t i = 0; i < rank; i++)
      dimensions[i] -= subtrahend[i];

    copy<typename O::value_type, rank>(&*result, &*first, dimensions, first.Strides(), result.Strides());
    first += dimensions;
    result += dimensions;
    return {std::move(first), std::move(result)};
  }

  template<std::ranges::input_range R, std::weakly_incrementable O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, O>
  constexpr std::ranges::copy_result<std::ranges::borrowed_iterator_t<R>, O>
  operator()(R&& r, O result) const
  {
    return (*this)(std::ranges::begin(r), std::ranges::end(r), std::move(result));
  }
};

template <typename TDevice> inline constexpr CopyFunc<TDevice> Copy;
template <typename TDevice> inline constexpr FillFunc<TDevice> Fill;



} // end of namespace grid

#endif  // GRID_TENSOR_BASE_TENSOR_H
