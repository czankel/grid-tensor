//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_UNARY_H
#define GRID_TENSOR_UNARY_H

#include <algorithm>
#include <ranges>
#include <span>

#include "concepts.h"
#include "tensor_operator.h"

namespace grid {

template <template <typename> typename, typename> class UnaryOperator;

/// @brief Unary is a wrapper for a device-specific unary operator implementation.
///
/// Unary provides a lazy-implementation that only stores the tensor and evaluates
/// the operation with operator().
///
/// Unary is typically not used directly, instead, use the actual functions, such as Neg(Tensor).
///
/// The actual operator implementations need to provide an operator() with an input and output
/// range. This differs from, e.g. std::ranges::transform that requires an output iterator instead
/// of a range.
///
///  template<std::ranges::input_range, std::ranges::output_range> operator();
///
///  @tparm TOperator unary operator type
///  @tparm TTensor  tensor type
///
template <typename TOperator, AnyTensor TTensor>
class Unary : public TensorOperator<typename std::remove_cvref_t<TTensor>::value_type,
                                    std::remove_cvref_t<TTensor>::rank,
                                    Unary<TOperator, TTensor>>
{
 public:
  using typename Unary::TensorOperator::value_type;
  using Unary::TensorOperator::rank;

  template <typename T>
  Unary(TOperator, T&& tensor)
    : TensorOperator<value_type, rank, Unary<TOperator, TTensor>>(*this),
      tensor_(std::forward<T>(tensor))
  {}

  ~Unary() {}

  Unary() = delete;
  Unary(const Unary& other) = delete;
  Unary& operator=(const Unary& other) = delete;

 public:

  /// operator()() evaluates the unary operator and returns a tensor.
  auto operator()() const
  {
    using ResultTensor = Tensor<value_type, rank, DeviceMemory<tensor_device_t<TTensor>>>;
    auto result = ResultTensor(tensor_.Dimensions(), Uninitialized<value_type>{});
    operator_(tensor_, result);
    return result;
  }

 private:
  static TOperator operator_;
  TTensor tensor_;
};

template <typename TOp, typename T> Unary(TOp, T&&) -> Unary<TOp, typename to_tensor<T>::type>;
template <typename TOperator, AnyTensor TTensor> TOperator Unary<TOperator, TTensor>::operator_;

//
// Elementary Unary Operators
//

template <typename> struct CopyOperator;
template <typename> struct NegOperator;

template <typename> class RmsNormOperator;
template <typename> class SoftMaxOperator;
template <typename> class SiluOperator;


/// @brief Copy returns a copy of the tensor.
template <TensorConvertible TTensor>
auto Copy(TTensor&& tensor)
{
  return Unary(UnaryOperator<CopyOperator, tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}

/// @brief Neg returns a copy of the negated tensor.
template <TensorConvertible TTensor>
auto Neg(TTensor&& tensor)
{
  return Unary(UnaryOperator<NegOperator, tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}

// FIXME: move to 'operators'; could use additional parameters, not really a unary??
/// @brief RmsNorm returns a tensor of the RMS normalized tensor.
template <TensorConvertible TTensor>
requires (std::remove_cvref_t<TTensor>::rank <= 2)
auto RmsNorm(TTensor&& tensor)
{
  return Unary(RmsNormOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}

/// @brief SoftMax returns a tensor with the SoftMax applied to the provided tensor.
template <TensorConvertible TTensor>
auto SoftMax(TTensor&& tensor)
{
  return Unary(SoftMaxOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}

/// @brief Silu returns a tensor with SiLU activation applied to the provided tensor.
template <TensorConvertible TTensor>
auto Silu(TTensor&& tensor)
{
  return Unary(SiluOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}

} // end of namespace grid

#endif  // GRID_TENSOR_UNARY_H
