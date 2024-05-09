//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_OPERATOR_H
#define GRID_TENSOR_TENSOR_OPERATOR_H

namespace grid {

namespace details {

/// @brief Helper function to reduce the rank in case of contiguous data for unary operators.
template <typename TOperator, size_t N>
void Fold(std::span<size_t,  N> dimensions,
          std::span<const ssize_t, N - 1> strides0,
          std::span<const ssize_t, N - 1> strides1,
          TOperator&& op)
{
  static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");

  if constexpr (N >= 2)
  {
    if (strides0[N - 2] - dimensions[N - 1] == 0 && strides1[N - 2] - dimensions[N - 1] == 0)
    {
      dimensions[N - 2] *=  dimensions[N - 1];
      Fold(dimensions.template first<N - 1>(),
           strides0.template first<N - 2>(),
           strides1.template first<N - 2>(),
           op);
      return;
    }
  }
  op(std::span<const size_t, N>(dimensions.begin(), N), std::move(strides0), std::move(strides1));
}

template <typename TOperator, size_t N>
void Fold(std::span<const size_t,  N> dimensions,
          std::span<const ssize_t, N - 1> strides0,
          std::span<const ssize_t, N - 1> strides1,
          TOperator&& op)
{
  static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");

  if constexpr (N >= 2)
  {
    if (strides0[N - 2] - dimensions[N - 1] == 0 && strides1[N - 2] - dimensions[N - 1] == 0)
    {
      std::array<size_t, N - 1> dim;
      std::ranges::copy(dimensions.template first<N - 1>(), dim.begin());
      dim[N - 2] *= dimensions[N - 1];

      Fold(std::span(dim),
           strides0.template first<N - 2>(),
           strides1.template first<N - 2>(),
           op);
      return;
    }
  }
  op(std::move(dimensions), std::move(strides0), std::move(strides1));
}

/// @brief Helper function to reduce the rank in case of contiguous data for binary operators.
template <typename TOperator, size_t N>
void Fold(std::span<size_t,  N> dimensions,
          std::span<const ssize_t, N - 1> strides0,
          std::span<const ssize_t, N - 1> strides1,
          std::span<const ssize_t, N - 1> strides2,
          TOperator&& op)
{
  static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");

  if constexpr (N >= 2)
  {
    if (strides0[N - 2] - dimensions[N - 1] == 0 &&
        strides1[N - 2] - dimensions[N - 1] == 0 &&
        strides2[N - 2] - dimensions[N - 1] == 0)
    {
      dimensions[N - 2] *=  dimensions[N - 1];
      Fold(dimensions.template first<N - 1>(),
           strides0.template first<N - 2>(),
           strides1.template first<N - 2>(),
           strides2.template first<N - 2>(),
           op);
      return;
    }
  }
  op(std::span<const size_t, N>(dimensions),
     std::move(strides0),
     std::move(strides1),
     std::move(strides2));
}

template <typename TOperator, size_t N>
void Fold(std::span<const size_t,  N> dimensions,
          std::span<const ssize_t, N - 1> strides0,
          std::span<const ssize_t, N - 1> strides1,
          std::span<const ssize_t, N - 1> strides2,
          TOperator&& op)
{
  static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");

  if constexpr (N >= 2)
  {
    if (strides0[N - 2] - dimensions[N - 1] == 0 &&
        strides1[N - 2] - dimensions[N - 1] == 0 &&
        strides2[N - 2] - dimensions[N - 1] == 0)
    {
      std::array<size_t, N - 1> dim;
      std::ranges::copy(dimensions.template first<N - 1>(), dim.begin());
      dim[N - 2] *= dimensions[N - 1];

      Fold(std::span(dim),
           strides0.template first<N - 2>(),
           strides1.template first<N - 2>(),
           strides2.template first<N - 2>(),
           op);
      return;
    }
  }
  op(std::move(dimensions),
     std::move(strides0),
     std::move(strides1),
     std::move(strides2));
}

} // end of namespace details


/// TensorOperator is a base class and wrapper for tensor operators.
///
/// Aliasing with partial specialization requires any CTAD rule to have a template-parameter-list
/// on the rhs that matches the aliasing definition This prohibites the use of more "fancy" rules,
/// such as defining the rank or type of a tensor.
///
/// The following rule, for example, would be ignored when aliasing with partial specialization
/// is used:
///
///   SomeClass(_args_) -> SomeClass<FancyType<_args_>>;
///
/// The TensorOperator exposes the necessary template parameters (value type and rank) to deduce
/// the same parameters in the Tensor class. This allowes derived operator classes to be used as
/// arguments in the Tensor constructor or assign operator.
///
/// @tparam T         value type
/// @tparam TRank     rank
/// @tparam TOperator wrapped operator type
template <typename T, size_t TRank, typename TOperator>
class TensorOperator
{
 public:
  using value_type = T;
  constexpr static size_t rank = TRank;

  /// Constructo
  /// @param op   Tensor operator object
  TensorOperator(TOperator& op) : op_(op) {}

  /// operator() executes the operation.
  auto operator()() const                             { return op_(); }

 private:
  TOperator& op_;
};

} // end of namespace grid

#endif	// GRID_TENSOR_TENSOR_OPERATOR_H
