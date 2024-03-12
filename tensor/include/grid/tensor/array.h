//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_ARRAY_H
#define GRID_TENSOR_ARRAY_H

#include "memory.h"

namespace grid {

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
