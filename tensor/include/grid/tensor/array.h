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

template <typename _Tp, size_t _Rank>
class ArrayView
{
 public:
  using value_type = _Tp;
  using pointer = _Tp*;
  using const_pointer = const _Tp*;

  /// Constructor for an array on a given data buffer with specified dimensions and strides.
  ArrayView(pointer data, const std::array<size_t, _Rank>&dimensions, const std::array<ssize_t, _Rank>&strides)
    : dimensions_(dimensions),
      strides_(strides),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(data)
  {}
  /// Constructor for an array on a given data buffer with specified dimensions and strides.
  ArrayView(pointer data, const size_t(&dimensions)[_Rank], const ssize_t(&strides)[_Rank])
    : dimensions_(get_array<size_t, _Rank>(dimensions)),
      strides_(get_array<ssize_t, _Rank>(strides)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(data)
  {}

  /// Constructor for an array on a given data buffer with specified dimensions.
  ArrayView(pointer data, const std::array<size_t, _Rank>& dimensions)
    : dimensions_(dimensions),
      strides_(make_strides<value_type>(dimensions_)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(data)
  {}


  /// Constructor for an array on a given data buffer with specified dimensions.
  ArrayView(pointer data, const size_t(&dimensions)[_Rank])
    : dimensions_(get_array<size_t, _Rank>(dimensions)),
      strides_(make_strides<value_type>(dimensions_)),
      size_(get_buffer_size(dimensions_, strides_)),
      data_(data)
  {}


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
  const_pointer Data() const                              { return data_; }

 private:
  std::array<size_t, _Rank>   dimensions_;
  std::array<ssize_t, _Rank>  strides_;
  size_t                      size_;
  pointer                     data_;
};

} // end of namespace grid

#endif // GRID_TENSOR_ARRAY_H
