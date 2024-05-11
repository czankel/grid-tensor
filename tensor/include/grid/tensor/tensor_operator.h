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

/// @brief Helper function to reduce the rank in case of contiguous data for binary operators.
///
/// The operator function is called with a boolean flag to indicate that the "lowest" dimension
/// is contiguous (strides are all implicit 1), and a new dimension span for the remaining
/// dimension or non-contiguous dimensions.
template <size_t TRank, typename TOp>
void Fold(TOp&& op, std::span<const size_t, TRank> dimensions, auto... strides)
{
  if constexpr (TRank < 1)
    op(std::span<const size_t, 0>(), false);

  else if constexpr (((strides.size() == 0) || ...))
    op(std::move(dimensions), false);

  else if constexpr (TRank == 1 || ((strides.size() == 1) || ...))
    op(std::move(dimensions), ((strides[0] == 1) && ...));

  // TRank > 1 from here on
  else if (((strides [TRank - 1] != 1) || ...))
    op(std::move(dimensions), false);

  else
  {
    static_assert(sizeof...(strides) > 1);
    constexpr size_t max_folds = std::min({strides.size()...});
    size_t fold_dim = dimensions[TRank - 1];

    auto foldfn = [&]<size_t I>() -> bool
    {
      if (I == max_folds - 1 || ((strides [strides .size() -I - 2] - fold_dim != 0) || ...))
      {
        std::array<size_t, TRank - I> dim;
        std::ranges::copy(dimensions.template first<TRank - I - 1>(), dim.begin());
        dim[TRank - I - 1] = fold_dim;
        op(std::span<const size_t, TRank - I>(dim.begin(), TRank - I), true);
        return false;
      }
      fold_dim *= dimensions[TRank - I - 2];
      return true;
    };

    [&] <std::size_t... I>(std::index_sequence<I...>)
    {
      (foldfn.template operator()<I>() && ...);
    }(std::make_index_sequence<max_folds>{});
  }
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
