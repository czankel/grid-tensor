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

namespace grid {

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


/// brief: Array is a specialization for a dynamically allocated buffer.
template <typename T>
class Array<T, DynamicMemory>
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

#endif  // GRID_TENSOR_BASE_ARRAY_H
