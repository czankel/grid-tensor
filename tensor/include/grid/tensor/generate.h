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


/// @brief GenerateFunction provides helper for applying an generator on a tensor.
///
/// GeneratemFunction is a class that provides overloaded operators operator() for
/// applying a generator on a range (Tensor) and stores the result in another provided
/// range (Tensor). The class shouldn't be used directly but through
/// the Generatefunction.
///
/// @see Generate()
///
/// Example:
///
///   Generate(tensor1, tensor2, result.begin(), BinaryOperator<AddOperator>{});
class GenerateFunction
{
 public:

  // generator-out
  template<std::input_or_output_iterator O, std::sentinel_for<O> S,
           std::copy_constructible F>
  constexpr std::ranges::generate_result<I, O>
  operator()(I first, S last, O result, F gen) const
  {
    constexpr size_t rank = O::rank;

    // TODO, identify if first is {0} and skip loop
    auto dimensions = last.Coordinates();
    auto& subtrahend = first.Coordinates();
    for (size_t i = 0; i < rank; i++)
      dimensions[i] -= subtrahend[i];

    std::invoke(gen, &*result, &*first, dimensions, result.Strides(), first.Strides());
    first += dimensions;
    result += dimensions;

    return {std::move(first), std::move(result)};
  }

  // in-out
  template<std::ranges::input_range R,
           std::weakly_incrementable O,
           std::copy_constructible F>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, O>
  constexpr std::ranges::unary_transform_result<std::ranges::borrowed_iterator_t<R>, O>
  operator()(R&& r, O result, F gen) const
  {
    return (*this)(std::ranges::begin(r), std::ranges::end(r),
                   std::move(result),
                   std::ref(gen));
  }
};


/// Generate applies the given generator to a provided range (Tensor)
inline constexpr GenerateFunction Generate;


} // end of namespace grid

#endif  // GRID_TENSOR_GENERATE_H
