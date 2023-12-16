//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_TENSOR_VIEW_H
#define GRID_TENSOR_BASE_TENSOR_VIEW_H

#include "copy.h"

namespace grid {

/// TensorView<Tensor, Rank> implements a view of a tensor.
///
/// Note that a view cannot be created from a temporary rval; it will return a tensor.
template <PrimitiveTensor TTensor, size_t TRank>
class TensorView
{
 public:
  using value_type = typename TTensor::value_type;
  using pointer = typename TTensor::pointer;
  using const_pointer = typename TTensor::const_pointer;
  constexpr static size_t rank = TRank;

 private:
  // pointer_cast provides a (value) const-qualified pointer cast
  template <typename T, typename S> requires (std::is_const_v<std::remove_pointer_t<S>>)
  const std::remove_pointer_t<T>* pointer_cast(S pointer)
  {
    return reinterpret_cast<const std::remove_pointer_t<T>*>(pointer);
  }
  template <typename T, typename S> requires (!std::is_const_v<std::remove_pointer_t<S>>)
  std::remove_pointer_t<T>* pointer_cast(S pointer)
  {
    return reinterpret_cast<std::remove_pointer_t<T>*>(pointer);
  }

 public:
  // TensorView must be in the same scope and lifetime as the underlying tensor.
  TensorView() = delete;

  /// Constructor
  template <size_t TTensorRank>
  explicit TensorView(TTensor& tensor, const ssize_t(& axes)[TRank], const ssize_t(& offsets)[TTensorRank])
    : size_(0UL)
  {
    std::bitset<TTensorRank> handled = false;
    auto strides = tensor.Strides();
    auto dimensions    = tensor.Dimensions();

    for (ssize_t i = static_cast<ssize_t>(TRank) - 1; i >= 0; i--)
    {
      if (axes[i] >= 0 && axes[i] < static_cast<ssize_t>(TTensorRank))
      {
        if (handled[axes[i]])
          throw std::runtime_error("axis can only be used once");

        handled.set(axes[i]);
        dimensions_[i] = dimensions[axes[i]];
        strides_[i] = strides[axes[i]];
        size_ += dimensions_[i] * strides_[i];
      }
      else if (axes[i] == Broadcast)
      {
        dimensions_[i] = 1;
        strides_[i] = 0;
      }
      else
        throw std::runtime_error("Invalid axis");
    }

    size_t offset = 0UL;
    for (size_t i = 0; i < TTensorRank; i++)
    {
      if (offsets[i] > static_cast<ssize_t>(tensor.dimensions_[i] * tensor.strides_[i]))
        throw std::runtime_error("Offset exceeds dimension");
      offset += offsets[i] * tensor.strides_[i];
    }

    data_ = reinterpret_cast<pointer>(pointer_cast<char*>(tensor.Data()) + offset);
  }


  /// operator=(Tensor) copies data from the rhs tensor (or view) into the view of the dependent tensor.
  template <AnyTensor TFromTensor> requires (TFromTensor::rank == TRank)
  auto operator=(const TFromTensor& rhs)
  {
    copy<value_type, TRank>(data_, rhs.Data(), dimensions_, strides_, rhs.Strides());
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return TRank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, TRank>& Dimensions() const     { return dimensions_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, TRank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return &data_; }
  const_pointer Data() const                              { return data_; }

 private:
  std::array<size_t, TRank>   dimensions_;
  std::array<ssize_t, TRank>  strides_;
  size_t                      size_;
  pointer                     data_;
};

} // end of namespace grid

#endif  // GRID_TENSOR_BASE_TENSOR_VIEW_H

