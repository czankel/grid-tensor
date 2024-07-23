//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_BASE_GENERATOR_H
#define GRID_TENSOR_BASE_GENERATOR_H

#include <span>
#include <algorithm>
#include <ranges>

namespace grid {

/// @brief GeneratorOperator provides the device specific implementation of a generator.
template <template <typename> typename TOperator>
class GeneratorOperator<TOperator<device::Base>>
{
  template <typename T, std::copy_constructible F>
  inline void generate(T* d,
                       std::span<const size_t, 1> dimensions,
                       std::span<const ssize_t, 1> strides,
                       F gen) const
  {
    for (size_t i = 0; i < dimensions[0]; i++, d += strides[0])
      TOperator<device::Base>()(d, gen);
  }

  template <typename T, size_t N, std::copy_constructible F>
  inline void generate(T* d,
                       std::span<const size_t, N> dimensions,
                       std::span<const ssize_t, N> strides,
                       F gen) const
  {
    for (size_t i = 0; i < dimensions[0]; i++, d += strides[0])
      generate(d,
               std::span<const size_t, N - 1>(dimensions.begin() + 1, dimensions.end()),
               std::span<const ssize_t, N - 1>(strides.begin() + 1, strides.end()),
               std::move(gen));
  }

 public:
  template<typename O, std::copy_constructible F>
  requires std::invocable<F&> && std::ranges::output_range<O, std::invoke_result_t<F&>>
  void operator()(O&& out, F gen) const
  {
    using tensor_type = std::remove_cvref_t<O>;
    constexpr size_t rank = tensor_type::rank;
    auto fist_d = std::ranges::begin(out);

    generate(&*fist_d, std::span<const size_t, rank>{fist_d.Extents()}, std::span{fist_d.Strides()}, std::move(gen));
  }
};

//
// Operators
//

template <> struct FillOperator<device::Base>
{
  template <typename T> inline void operator()(T* d, const T val) const { *d = val; }
};

template <> struct FunctionOperator<device::Base>
{
  // FIXME requires std::invocable<F&> && std::indirectly_writable<O, std::invoke_result_t<F&>>
  template <typename T, typename F> inline void operator()(T* d, F&& gen) const { *d = std::invoke(gen); }
};


} // end of namespace grid

#endif  // GRID_TENSOR_BASE_GENERATOR_H
