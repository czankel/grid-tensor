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

namespace {

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

} // end of namespace


/// TensorView<Tensor, Rank> implements a view of a tensor.
///
/// Note that a view cannot be created from a temporary rval; it will return a tensor.
template <PrimitiveTensor _Tensor, size_t _ViewRank>
class TensorView
{
 public:
  using value_type = typename _Tensor::value_type;
  using pointer = typename _Tensor::pointer;
  using const_pointer = typename _Tensor::const_pointer;
  constexpr static size_t rank = _ViewRank;

  // FIXME: maybe allow?
  // TensorView must be in the same scope and lifetime as the underlying tensor.
  TensorView() = default;


  /// Constructor with arguments
  explicit TensorView(_Tensor& tensor,
                      const std::array<size_t,  _ViewRank>& dimensions,
                      const std::array<ssize_t, _ViewRank>& strides,
                      size_t size,
                      size_t offset)
    : tensor_(tensor),
      dimensions_(dimensions),
      strides_(strides),
      size_(size),
      data_(reinterpret_cast<pointer>(pointer_cast<char*>(tensor.Data()) + offset))
  {
    //printf("TensorView tensor %p offset %lu -> %p\n", tensor.Data(), offset, data_);
  }


  // FIXME: copy constructor? move constructor?

  /// operator=(Tensor) copies data from the rhs tensor (or view) into the view of the dependent tensor.
  template <AnyTensor _FromTensor> requires (_FromTensor::rank == _ViewRank)
  auto operator=(const _FromTensor& rhs)
  {
    //printf("View copy %p -> %p\n", rhs.Data(), data_);
    copy<value_type, _ViewRank>(data_, rhs.Data(), dimensions_, strides_, rhs.Strides());
  }

  template <AnyOperator _Operator> // FIXME requires PrimitiveTensor<to_tensor(_Operator)>
  auto operator=(const _Operator& oper)
  {
    return this->operator=(oper());
  }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return _ViewRank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, _ViewRank>& Dimensions() const { return dimensions_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, _ViewRank>& Strides() const   { return strides_; }

  /// Size returns the data buffer size.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return data_; }
  const_pointer Data() const                              { return data_; }

 private:
  _Tensor&                        tensor_;
  std::array<size_t, _ViewRank>   dimensions_;
  std::array<ssize_t, _ViewRank>  strides_;
  size_t                          size_;
  pointer                         data_;
};

namespace view {

/// Span returns a view of a subspace of the provide tensor with reduced dimensions
/// but keeping ... FIXME
template <typename _Tensor>
inline auto Span(_Tensor& tensor, const std::array<size_t, _Tensor::rank>& dimensions)
{
  auto strides = make_strides(dimensions);
  return TensorView(tensor, dimensions, strides, get_buffer_size(dimensions, strides), 0);
}

template <typename _Tensor>
auto Span(_Tensor& tensor,
          const std::array<size_t, _Tensor::rank>& dimensions,
          const std::array<size_t, _Tensor::rank>& offsets)
{
  auto tensor_dimensions = tensor.Dimensions();
  auto strides = tensor.Strides();
  size_t offset = 0UL;
#if 0
  constexpr bool is_zero = [=] {
    for (size_t i = 0; i < _Tensor::rank; i++)
      if constexpr (offsets[i] != 0)
        return false;
    return true;
  }();

  if constexpr (!is_zero)
  {
#endif
    for (size_t i = 0; i < _Tensor::rank; i++)
    {
      if (dimensions[i] == 1 && offsets[i] > 0)
        throw std::runtime_error("invalid dimension on broadcast");
      if (dimensions[i] + offsets[i] > tensor_dimensions[i])
        throw std::runtime_error("dimensions and offset exceeds tensor dimension: " +
            std::to_string(dimensions[i]+offsets[i]) + " > " + std::to_string(tensor_dimensions[i]));
      offset += offsets[i] * strides[i];
    }
  //}

  return TensorView(tensor, dimensions, strides, get_buffer_size(dimensions, strides), offset);
}

// FIXME: if n-rank tensor, Rearrange is easy to make things wrong: Rearrange({x}, {y}} --> y is for all ranks!
//
/// Rearrange returns a view of the provided tensor with changed axes supporting also broadcast.
/// @param axes  Use information from the specified axis in the tensor. Use Broadcast for broadcast.
/// @param dimensions  Updated dimensions. Use 0 to keep original dimension
/// @param offsets  Offsets in the original tensor dimensions
template <typename _Tensor, size_t _ViewRank>
auto Rearrange(_Tensor& tensor,
               const std::array<ssize_t, _ViewRank>& axes,
               const std::array<size_t, _ViewRank>& dimensions,
               const std::array<size_t, _Tensor::rank>& offsets)
{
  std::bitset<_Tensor::rank> handled = false;
  auto tensor_dimensions = tensor.Dimensions();
  auto tensor_strides = tensor.Strides();

  std::array<size_t, _ViewRank> view_dimensions;
  std::array<ssize_t, _ViewRank> view_strides;

  size_t size = 0;

  for (ssize_t i = static_cast<ssize_t>(_ViewRank) - 1; i >= 0; i--)
  {
    ssize_t axis = axes[i];
    if (axis >= 0 && axis < static_cast<ssize_t>(_Tensor::rank))
    {
      if (handled[axis])
        throw std::runtime_error("axis can only be used once");

      handled.set(axis);
      // FIXME: assert on dimension <= tensor_dimension
      view_dimensions[i] = dimensions[i] != 0? dimensions[i] : tensor_dimensions[axis];
      view_strides[i] = tensor_strides[axis];

      size = std::max(size, view_dimensions[i] * view_strides[i]);
    }
    else if (axis == Broadcast)
    {
      view_dimensions[i] = 1;
      view_strides[i] = 0;
    }
    else
      throw std::runtime_error(std::string("Invalid axis: ") + std::to_string(axis));
  }

  size_t offset = 0;
  for (size_t i = 0; i < _Tensor::rank; i++)
  {
    /* FIXME
       if (indices[i] > static_cast<size_t>(tensor_dimensions[i] * tensor_strides[i]))
       throw std::runtime_error("Offset exceeds dimension");
       */
    // FIXME no offset on broadcast allowed
    offset += offsets[i] * tensor_strides[i];
  }

  return TensorView(tensor, view_dimensions, view_strides, size, offset);
}

// FIXME -- same as TensorView
/// Reshape returns a view of the provide tensor with ...
template <typename _Tensor>
auto Reshape(const _Tensor& tensor,
             const std::array<size_t, _Tensor::rank>& dimensions,
             size_t offset = 0)
{
  auto strides = make_strides(dimensions);
  size_t size = get_buffer_size(dimensions, strides);
  // assert orig-size >= size + offset
  return TensorView(tensor, dimensions, strides, size, offset);
}
template <typename _Tensor>
auto Reshape(const _Tensor& tensor,
             const std::array<size_t, _Tensor::rank>& dimensions,
             const std::array<ssize_t, _Tensor::rank>& strides,
             size_t offset = 0)
{
  size_t size = get_buffer_size(dimensions, strides);
  // assert orig-size >= size + offset
  return TensorView(tensor, dimensions, strides, size, offset);
}

} // end of namespace view
} // end of namespace grid


// FIXME: remove TensorView
#if 0
template <PrimitiveTensor _Tensor, size_t _Rank>
explicit TensorView(const std::array<size_t, _Rank>&, const std::array<ssize_t, _Rank>&,
                    size_t, typename _Tensor::pointer)
  -> TensorView<_Tensor, _Rank>;
template <PrimitiveTensor _Tensor, size_t _Rank>
explicit TensorView(_Tensor&, const std::array<size_t, _Rank>&, const std::array<ssize_t, _Rank>&, size_t)
  -> TensorView<_Tensor, _Rank>;
template <PrimitiveTensor _Tensor, size_t _Rank>
  -> TensorView<_Tensor, _Rank>;
template <PrimitiveTensor _Tensor, size_t _Rank>
explicit TensorView(_Tensor&, const std::array<size_t, _Rank>&, size_t)
  -> TensorView<_Tensor, _Rank>;
template <PrimitiveTensor _Tensor, size_t _Rank>
explicit TensorView(_Tensor&, const std::array<size_t, _Rank>&)
  -> TensorView<_Tensor, _Rank>;
#endif

#endif  // GRID_TENSOR_BASE_TENSOR_VIEW_H
