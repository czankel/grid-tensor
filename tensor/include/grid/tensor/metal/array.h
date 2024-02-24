//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_METAL_ARRAY_H
#define GRID_TENSOR_METAL_ARRAY_H

#include "device.h"

namespace grid {

/// brief: Array is a specialization for a dynamically allocated buffer.
template <typename T>
class Array<T, DynamicMemory<device::Metal>>
{
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;


  inline MTL::Buffer* Allocate(size_t size)
  {
    // Align up memory
    if (size > vm_page_size)
      size = vm_page_size * ((size + vm_page_size - 1) / vm_page_size);

    // TODO: cache buffers, check 'memory pressure' and cause "wait"

    // Allocate new buffer
    size_t mode = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;
    auto device = device::Metal::GetDevice();
    return device.NewBuffer(size, mode);
  }

  inline void Free(MTL::Buffer* buffer)
  {
    buffer->release();
  }


 public:
  Array() = default;

  // Explicity disallow copy construction as Array isn't fully aware of any buffer structure.
  Array(const Array& other) = delete;

  // @brief Move constructor.
  Array(Array&& other) : size_(other.size_), buffer_(std::move(other.buffer_)) { other.buffer_ = nullptr; }

  // @brief Allocates a buffer of the provided size.
  Array(size_t size) : size_(size)
  {
    buffer_ = Allocate(size_);
  }

  ~Array()
  {
    if (buffer_ != nullptr)
      Free(buffer_);
  }

  Array& operator=(Array&& other)
  {
    if (buffer_ != nullptr)
      Free(buffer_);

    size_ = other.size_;
    buffer_ = std::move(other.buffer_);
    other.buffer_ = nullptr;

    return *this;
  }

  Array& operator=(const Array& other) = delete;


  /// Resize resizes the buffer of the Array. This will destroy
  Array& Realloc(size_t size)
  {
    if (size != size_)
    {
      if (buffer_ != nullptr)
        Free(buffer_);
      buffer_ = Allocate(device::Metal::GetDevice(), size);
    }

    return *this;
  }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return static_cast<pointer>(buffer_->contents()); }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return static_cast<const_pointer>(buffer_->contents()); }

 protected:
  size_t  size_;
  MTL::Buffer* buffer_;
};

} // end of namespace grid

#endif  // GRID_TENSOR_METAL_ARRAY_H
