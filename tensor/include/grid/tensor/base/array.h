//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_ARRAY_H
#define GRID_TENSOR_BASE_ARRAY_H

#include "device.h"

namespace grid {

/// brief: Array is a specialization for a dynamically allocated buffer.
template <typename T>
class Array<T, DeviceMemory<device::Base>>
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


} // end of namespace grid

#endif  // GRID_TENSOR_BASE_ARRAY_H
