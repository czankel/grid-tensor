//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_ARRAY_H
#define GRID_TENSOR_ARRAY_H

#include <span>

#include "memory.h"

namespace grid {

namespace details {

// copy copies the data between buffers accordig to dimensions and strides.
template <typename T>
inline void
copy(T* dst, const T* src,
     std::span<const size_t,  0>,
     std::span<const ssize_t, 0>,
     std::span<const ssize_t, 0>)
{
  *dst = *src;
}

template <typename T>
inline void
copy(T* dst, const T* src,
     std::span<const size_t,  1> dimensions,
     std::span<const ssize_t, 1> strides1,
     std::span<const ssize_t, 1> strides2)
{
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    *dst = *src;
    dst += strides1[0];
    src += strides2[0];
  }
}

template <typename T, size_t N>
inline std::enable_if_t<(N > 1), void>
copy(T* dst, const T* src,
     std::span<const size_t,  N> dimensions,
     std::span<const ssize_t, N> strides1,
     std::span<const ssize_t, N> strides2)
{
  static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    copy(dst, src,
         std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
         std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1),
         std::span<const ssize_t, N - 1>(strides2.begin() + 1, N - 1));
    dst += strides1[0];
    src += strides2[0];
  }
}

template <typename T>
inline void initialize(T* dst, std::span<size_t, 1> dimensions, std::span<ssize_t, 1> strides, T init)
{
  for (size_t i = 0; i < dimensions[0]; i++, reinterpret_cast<char*&>(dst) += strides[0])
    *dst = init;
}

template <typename T, size_t N>
inline void initialize(T* dst, std::span<size_t, N> dimensions, std::span<ssize_t, N> strides, T init)
{
  for (size_t i = 0; i < dimensions[0]; i++, reinterpret_cast<char*&>(dst) += strides[0])
    initialize(dst,
        std::span<size_t, N - 1>(dimensions.begin() + 1, dimensions.end()),
        std::span<ssize_t, N - 1>(strides.begin() + 1, strides.end()),
        init);
}

template <typename T>
inline void initialize(T* dst, size_t size, T init)
{
  for (size_t i = 0; i < size; i++)
    *dst++ = init;
}

} // end of namespace details

/// Array manages a buffer of elements of a specific type.
///
/// The buffer can be statically or dynamically allocated, and in system memory or device memory.
template <typename, typename> class Array;


/// Array specialization for storing a single scalar
template <typename T>
class Array<T, Scalar>
{
 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;

 public:
  Array() = default;

  // @brief Initializes the data value to init.
  Array(value_type init) : data_(init) {}

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type); }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return &data_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return &data_; }

 protected:
  value_type  data_;
};


/// Array specialization for static data.
template <typename T, size_t... Ns>
class Array<T, StaticMemory<Ns...>>
{
 public:
  using value_type = T;
  using pointer = const value_type*;
  using const_pointer = const value_type*;
  static constexpr size_t size = (... * Ns);


 public:
  Array() = default;

  // Explicity disallow copy construction as Array isn't fully aware of any buffer structure.
  Array(const Array& other) = delete;

  // @brief Move constructor.
  Array(Array&& other) = delete; // : size_(other.size_), data_(std::move(other.data_)) { other.data_ = nullptr; }

  // @brief Allocates a buffer of the provided size. FIXME
  Array(std::array<T, size>&& array) : array_(array) {}
  Array& operator=(Array&& other) = delete;
  Array& operator=(const Array& other) = delete;


  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return sizeof(value_type) * size; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return array_.data(); }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }

 protected:
  std::array<value_type, size>  array_;
};


} // end of namespace grid

#endif // GRID_TENSOR_ARRAY_H
