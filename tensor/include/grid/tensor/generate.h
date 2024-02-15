//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_GENERATE_H
#define GRID_TENSOR_GENERATE_H

#include <span>
#include <algorithm>
#include <ranges>

#include "concepts.h"

namespace grid {


/// @brief GenerateFunction provides helper for applying a generator on a tensor.
///
/// GenerateFunction is a class that provides overloaded operator() for applying a
/// generator on a range (Tensor).
///
/// The class shouldn't be used directly but through the Generatefunction.
///
/// @see Generate()
///
/// Example:
///
///   Generate(tensor, [n = 0] { return n++; });
class GenerateFunction
{
  template <typename T, std::copy_constructible F>
  inline void generate(T* dst,
                       std::span<const size_t, 1> dimensions,
                       std::span<const ssize_t, 1> strides,
                       F gen) const
  {
    for (size_t i = 0; i < dimensions[0]; i++, dst += strides[0])
      *dst = std::invoke(gen);
  }

  template <typename T, size_t N, std::copy_constructible F>
  inline void generate(T* dst,
                       std::span<const size_t, N> dimensions,
                       std::span<const ssize_t, N> strides,
                       F gen) const
  {
    for (size_t i = 0; i < dimensions[0]; i++, dst += strides[0])
      generate(dst,
               std::span<const size_t, N - 1>(dimensions.begin() + 1, dimensions.end()),
               std::span<const ssize_t, N - 1>(strides.begin() + 1, strides.end()),
               std::move(gen));
  }


 public:
  template<std::input_or_output_iterator O, std::sentinel_for<O> S, std::copy_constructible F>
  requires std::invocable<F&> && std::indirectly_writable<O, std::invoke_result_t<F&>>
  constexpr O operator()(O first, S last, F gen) const
  {
    constexpr size_t rank = O::rank;

    // TODO, identify if first is {0} and skip loop
    auto dimensions = last.Coordinates();
    auto& subtrahend = first.Coordinates();
    for (size_t i = 0; i < rank; i++)
      dimensions[i] -= subtrahend[i];

    // TODO: support implementation optimizations in provided generator: if constexpr (generator_has_extra_args) std::invoke(gen, &*first, dimensions, first.Strides());
    generate(&*first, std::span<const size_t, rank>{dimensions}, std::span{first.Strides()}, std::move(gen));

    first += dimensions;

    return std::move(first);
  }

  template<class R, std::copy_constructible F>
  requires std::invocable<F&> && std::ranges::output_range<R, std::invoke_result_t<F&>>
  constexpr std::ranges::borrowed_iterator_t<R> operator()(R&& r, F gen) const
  {
    return (*this)(std::ranges::begin(r), std::ranges::end(r), std::move(gen));
  }
};


/// Generate applies the given generator to a provided range (Tensor)
inline constexpr GenerateFunction Generate;

} // end of namespace grid

#endif  // GRID_TENSOR_GENERATE_H
