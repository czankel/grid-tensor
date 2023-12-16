//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_ARRAY_H
#define GRID_TENSOR_ARRAY_H

#include "tensor_parameters.h"

namespace grid {

template <typename T, size_t TRank>
class ArrayView
{
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;

  /// Constructor for an array on a given data buffer with specified dimensions and strides.
  ArrayView(pointer data, const std::array<size_t, TRank>&dimensions, const std::array<ssize_t, TRank>&strides)
    : dimensions_(dimensions),
      strides_(strides),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(data)
  {}
  /// Constructor for an array on a given data buffer with specified dimensions and strides.
  ArrayView(pointer data, const size_t(&dimensions)[TRank], const ssize_t(&strides)[TRank])
    : dimensions_(get_array<size_t, TRank>(dimensions)),
      strides_(get_array<ssize_t, TRank>(strides)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(data)
  {}

  /// Constructor for an array on a given data buffer with specified dimensions.
  ArrayView(pointer data, const std::array<size_t, TRank>& dimensions)
    : dimensions_(dimensions),
      strides_(make_strides<value_type>(dimensions_)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(data)
  {}


  /// Constructor for an array on a given data buffer with specified dimensions.
  ArrayView(pointer data, const size_t(&dimensions)[TRank])
    : dimensions_(get_array<size_t, TRank>(dimensions)),
      strides_(make_strides<value_type>(dimensions_)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(data)
  {}


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return TRank; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, TRank>& Dimensions() const     { return dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, TRank>& Strides() const       { return strides_; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return data_; }
  const_pointer Data() const                              { return data_; }

 private:
  std::array<size_t, TRank>   dimensions_;
  std::array<ssize_t, TRank>  strides_;
  size_t                      size_;
  pointer                     data_;
};

} // end of namespace grid

#endif // GRID_TENSOR_ARRAY_H
