//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_TENSOR_ITERAOR_H
#define GRID_TENSOR_BASE_TENSOR_ITERAOR_H

namespace grid {
namespace details {

class Tensor;

// TODO: ConstIterator is mostly a duplicate for Iterator but for const_pointer

/// Iterator implements a bidirection iterator for tensors.
template <typename TTensor>
class Iterator
{
 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = typename TTensor::value_type;
  using pointer           = typename TTensor::pointer;
  using reference         = typename TTensor::reference;
  static constexpr size_t rank = TTensor::rank;

  Iterator() = default;

  // Note that ranges iterators must be default-initializable, so cannot use reference for Tensor
  Iterator(const TTensor* tensor, pointer data) : coordinates_{0}, ptr_(data), data_(data), tensor_(tensor) {}

  // FIXME: dimensions are a position and the position must be added to ptr_
  Iterator(const TTensor* tensor, pointer data, const std::array<size_t, rank>& dimensions)
    : coordinates_(dimensions),
      ptr_(data),
      data_(data),
      tensor_(tensor)
  {}

  reference operator*() const                     { return *ptr_; }

  // TODO: unrol loops?
  Iterator& operator++()
  {
    auto& extents = tensor_->Dimensions();
    auto& strides = tensor_->Strides();

    ptr_++;
    size_t index = 0;
    for (; index < rank && ++coordinates_[index] == extents[index]; index++)
      coordinates_[index] = 0;
    // non-contiguous, note that all coordinate's so far are 0
    if (index > 0)
      for (ptr_ = data_; index < rank; index++)
        reinterpret_cast<char*&>(ptr_) += coordinates_[index] * strides[index];

    return *this;
  }

  Iterator& operator--()
  {
    auto& extents = tensor_->Dimensions();
    auto& strides = tensor_->Strides();

    // TODO: assert > data_
    ptr_--;
    size_t index = 0;
    for (; index < rank && --coordinates_[index] == 0; index++)
      coordinates_[index] = extents[index];
    // non-contiguous, note that all coordinate's so far are 0
    if (index > 0)
      for (ptr_ = data_; index < rank; index++)
        reinterpret_cast<char*&>(ptr_) += coordinates_[index] * strides[index];

    return *this;
  }

  Iterator operator++(int)                        { Iterator tmp = *this; ++(*this); return tmp; }
  Iterator operator--(int)                        { Iterator tmp = *this; --(*this); return tmp; }

  /// @brief return a new iterator with added distances to the positions for the provided dimensions.
  template <size_t S>
  Iterator operator+(const std::array<size_t, S>& dist)
  {
    Iterator tmp = *this;
    for (size_t i = 0; i < S; i++)
      tmp.coordinates_[i] += dist[i];
    return tmp;
  }

  /// @brief add distances to the positions of the iterator for the provided dimensions.
  template <size_t S>
  Iterator& operator+=(const std::array<size_t, S>& dist)
  {
    for (size_t i = 0; i < S; i++)
      coordinates_[i] += dist[i];
    return *this;
  }

  friend bool operator==(const Iterator& a, const Iterator& b) { return a.ptr_ == b.ptr_; }

  // iterator extensions
  size_t                            Rank()              { return rank; }
  const std::array<size_t, rank>&   Extents() const     { return tensor_->Dimensions(); }
  const std::array<ssize_t, rank>&  Strides() const     { return tensor_->Strides(); }
  const std::array<size_t, rank>&   Coordinates() const { return coordinates_; }

 private:
  std::array<size_t, rank>  coordinates_;
  pointer                   ptr_;
  pointer                   data_;
  const TTensor*            tensor_;
};


/// ConstIterator implements a bidirection iterator for tensors.
template <typename TTensor>
class ConstIterator
{
 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = typename TTensor::value_type;
  using const_pointer     = typename TTensor::const_pointer;
  using const_reference   = typename TTensor::const_reference;
  static constexpr size_t rank = TTensor::rank;

  ConstIterator() = default;

  // Note that ranges iterators must be default-initializable, so cannot use reference for Tensor
  ConstIterator(const TTensor* tensor, const_pointer data)
  requires (rank == 0)
    : ptr_(data),
      data_(data),
      tensor_(tensor)
  {}

  ConstIterator(const TTensor* tensor, const_pointer data)
    : coordinates_{0},
      ptr_(data),
      data_(data),
      tensor_(tensor)
  {}

  // FIXME: dimensions are a position and the position must be added to ptr_
  ConstIterator(const TTensor* tensor, const_pointer data, const std::array<size_t, rank>& dimensions)
    : coordinates_(dimensions),
      ptr_(data),
      data_(data),
      tensor_(tensor)
  {}

  const_reference operator*() const                { return *ptr_; }

  // TODO: unrol loops?
  ConstIterator& operator++()
  {
    auto& extents = tensor_->Dimensions();
    auto& strides = tensor_->Strides();

    ptr_++;
    size_t index = 0;
    for (; index < rank && ++coordinates_[index] == extents[index]; index++)
      coordinates_[index] = 0;
    // non-contiguous, note that all coordinate's so far are 0
    if (index > 0)
      for (ptr_ = data_; index < rank; index++)
        reinterpret_cast<char*&>(ptr_) += coordinates_[index] * strides[index];

    return *this;
  }

  ConstIterator& operator--()
  {
    auto& extents = tensor_->Dimensions();
    auto& strides = tensor_->Strides();

    // TODO: assert > data_
    ptr_--;
    size_t index = 0;
    for (; index < rank && --coordinates_[index] == 0; index++)
      coordinates_[index] = extents[index];
    // non-contiguous, note that all coordinate's so far are 0
    if (index > 0)
      for (ptr_ = data_; index < rank; index++)
        reinterpret_cast<char*&>(ptr_) += coordinates_[index] * strides[index];

    return *this;
  }

  ConstIterator operator++(int)                  { ConstIterator tmp = *this; ++(*this); return tmp; }
  ConstIterator operator--(int)                  { ConstIterator tmp = *this; --(*this); return tmp; }

  /// @brief return a new iterator with added distances to the positions for the provided dimensions.
  template <size_t S>
  ConstIterator operator+(const std::array<size_t, S>& dist)
  {
    ConstIterator tmp = *this;
    for (size_t i = 0; i < S; i++)
      tmp.coordinates_[i] += dist[i];
    return tmp;
  }

  /// @brief add distances to the positions of the iterator for the provided dimensions.
  template <size_t S>
  ConstIterator& operator+=(const std::array<size_t, S>& dist)
  {
    for (size_t i = 0; i < S; i++)
      coordinates_[i] += dist[i];
    return *this;
  }

  friend bool operator==(const ConstIterator& a, const ConstIterator& b) { return a.ptr_ == b.ptr_; }

  // iterator extensions
  size_t                            Rank()              { return rank; }
  const std::array<size_t, rank>&   Extents() const     { return tensor_->Dimensions(); }
  const std::array<ssize_t, rank>&  Strides() const     { return tensor_->Strides(); }
  const std::array<size_t, rank>&   Coordinates() const { return coordinates_; }

 private:
  std::array<size_t, rank>  coordinates_;
  const_pointer             ptr_;
  const_pointer             data_;
  const TTensor*            tensor_;
};

//  ambiguous deduction in CLANG https://cplusplus.github.io/CWG/issues/2628.html
#ifdef __clang__
template <typename TTensor> requires (std::remove_cvref_t<TTensor>::rank == 0)
ConstIterator(const TTensor* tensor, typename std::remove_cvref_t<TTensor>::const_pointer data) -> ConstIterator<TTensor>;
template <typename TTensor>
ConstIterator(const TTensor* tensor, typename std::remove_cvref_t<TTensor>::const_pointer data) -> ConstIterator<TTensor>;
#endif

} // end of namespace details
} // end of namespace grid

#endif  // GRID_TENSOR_BASE_TENSOR_ITERAOR_H
