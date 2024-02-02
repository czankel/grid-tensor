//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_TENSOR_H
#define GRID_TENSOR_BASE_TENSOR_H

#include <array>
#include <bitset>
#include <initializer_list>
#include <numeric>
#include <tuple>

#include "../device.h"
#include "../tensor_parameters.h"

#include "array.h"
#include "copy.h"

namespace grid {


namespace {

template <typename T>
inline void initialize(T* dst, std::span<const size_t, 1> dimensions, std::span<const ssize_t, 1> strides, const T& init)
{
  for (size_t i = 0; i < dimensions[0]; i++, dst += strides[0])
    *dst = init;
}

template <typename T, size_t N>
inline void initialize(T* dst, std::span<const size_t, N> dimensions, std::span<const ssize_t, N> strides, const T& init)
{
  for (size_t i = 0; i < dimensions[0]; i++, dst += strides[0])
    initialize(dst,
        std::span<const size_t, N - 1>(dimensions.begin() + 1, dimensions.end()),
        std::span<const ssize_t, N - 1>(strides.begin() + 1, strides.end()),
        init);
}

} // end of namespace details


// FIXME: the coordinates of the first tensor must be added to the
template <>
struct FillFunc<device::CPU>
{
  template<typename T, std::output_iterator<const T&> O, std::sentinel_for<O> S>
  constexpr O operator()(O first, S last, const T& value) const
  {
    constexpr size_t rank = O::rank;

    // TODO, identify if first is {0} and skip loop
    auto dimensions = last.Coordinates();
    auto& subtrahend = first.Coordinates();
    for (size_t i = 0; i < rank; i++)
      dimensions[i] -= subtrahend[i];

    initialize(&*first, std::span<const size_t, rank>{dimensions}, std::span{first.Strides()}, value);
    return first;
  }

  template<typename T, std::ranges::output_range<const T&> R>
  constexpr std::ranges::borrowed_iterator_t<R> operator()(R&& r, const T& value) const
  {
    return (*this)(std::ranges::begin(r), std::ranges::end(r), value);
  }
};



template <>
struct CopyFunc<device::CPU>
{
  template<std::input_iterator I, std::sentinel_for<I> S, std::weakly_incrementable O>
  requires std::indirectly_copyable<I, O>
  constexpr std::ranges::copy_result<I, O> operator()(I first, S last, O result) const
  {
    constexpr size_t rank = O::rank;

    // TODO, identify if first is {0} and skip loop
    auto dimensions = last.Coordinates();
    auto& subtrahend = first.Coordinates();
    for (size_t i = 0; i < rank; i++)
      dimensions[i] -= subtrahend[i];

    copy<typename O::value_type, rank>(&*result, &*first, dimensions, first.Strides(), result.Strides());
    first += dimensions;
    result += dimensions;
    return {std::move(first), std::move(result)};
  }

  template<std::ranges::input_range R, std::weakly_incrementable O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, O>
  constexpr std::ranges::copy_result<std::ranges::borrowed_iterator_t<R>, O>
  operator()(R&& r, O result) const
  {
    return (*this)(std::ranges::begin(r), std::ranges::end(r), std::move(result));
  }
};

template <typename TDevice> inline constexpr CopyFunc<TDevice> Copy;
template <typename TDevice> inline constexpr FillFunc<TDevice> Fill;



} // end of namespace grid

#endif  // GRID_TENSOR_BASE_TENSOR_H
