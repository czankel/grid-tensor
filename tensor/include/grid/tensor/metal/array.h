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

#include "../array.h"
#include "../tensor_parameters.h"

namespace grid {

/// brief: Array is a specialization for a dynamically allocated buffer.
template <typename T>
class Array<T, DeviceMemory<device::Metal>>
{
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;


  // TODO cache buffers, check 'memory pressure' and cause "wait", etc.
  inline MTL::Buffer* Allocate(size_t size)
  {
    // Align up memory
    // TODO only for larger sizes not < page_size?
    if (size > vm_page_size)
      size = vm_page_size * ((size + vm_page_size - 1) / vm_page_size);

    // Allocate new buffer
    size_t mode = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;
    auto& device = device::Metal::GetDevice();
    return device.NewBuffer(size, mode);
  }

  inline void Free(MTL::Buffer* buffer)
  {
    buffer->release();
  }


 public:
  Array() = default;

  // @brief Constructor for a contiguous array with the provided size.
  Array(size_t size) : size_(size), buffer_(Allocate(size)) {}

  // @brief Constructor for a contiguous array with the provided size with initialization.
  Array(size_t size, value_type init) : size_(size), buffer_(Allocate(size))
  {
    details::initialize(Data(), size_, init);
  }

  // @brief Constructor for a non-contiguous array with the provided dimensions and strides.
  template <size_t N>
  Array(const std::array<size_t, N>& dimensions, const std::array<ssize_t, N>& strides)
    : size_(get_buffer_size<value_type>(dimensions, strides)),
      buffer_(Allocate(size_))
  {}

  // @brief Constructor for a non-contiguous array with the provided dimensions and strides with initialization.
  template <size_t N>
  Array(const std::array<size_t, N>& dimensions, const std::array<ssize_t, N>& strides, value_type init)
    : size_(get_buffer_size<value_type>(dimensions, strides)),
      buffer_(Allocate(size_))
  {
    details::initialize(Data(), dimensions, strides, init);
  }


  // @brief Copy constructor of contiguous arrays.
  Array(const Array& other)
    : size_(other.size_),
      buffer_(Allocate(size_))
  {
    // TODO use GPU? Handle mmap, etc.
    memcpy(Data(), other.Data(), other.size);
  }

  // @brief Copy constructor with dimensions and strides
  template <size_t N>
  Array(const_pointer data,
        const std::array<size_t, N>& dimensions,
        const std::array<ssize_t, N>& strides1,
        const std::array<ssize_t, N>& strides2)
    : size_(get_buffer_size<value_type>(dimensions, strides1)),
      buffer_(Allocate(size_))
  {
    details::copy(Data(), data,
                  std::span<const size_t, N>(dimensions.begin(), N),
                  std::span<const ssize_t, N>(strides1.begin(), N),
                  std::span<const ssize_t, N>(strides2.begin(), N));
  }

  // @brief Move constructor.
  Array(Array&& other) : size_(other.size_), buffer_(std::move(other.buffer_)) { other.buffer_ = nullptr; }


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

  // TODO: Implement copy. Is Copy actually needed?
#if 0
  template <size_t N>
  Array& Copy(const Array& other,
              const std::array<size_t, N>& dimensions,
              const std::array<ssize_t, N>& strides1,
              const std::array<ssize_t, N>& strides2)
  {
    Realloc(size_);
  }
#endif

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return static_cast<pointer>(buffer_->contents()); }

  /// Data returns a const_pointer to the data buffer.
  const_pointer Data() const                              { return static_cast<const_pointer>(buffer_->contents()); }

  // Buffer returns the MTL buffer - internal use only
  MTL::Buffer* Buffer()                                    { return buffer_; }

  // Buffer returns the MTL buffer - internal use only
  const MTL::Buffer* Buffer() const                        { return buffer_; }

 protected:
  size_t  size_;
  MTL::Buffer* buffer_;
};

} // end of namespace grid

#endif  // GRID_TENSOR_METAL_ARRAY_H
