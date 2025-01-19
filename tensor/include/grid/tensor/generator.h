//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_GENERATOR_H
#define GRID_TENSOR_GENERATOR_H

namespace grid {

template <typename> class GeneratorOperator;

template <typename> struct FillOperator;
template <typename> struct FunctorOperator;

/// @brief Generator applies the given generator operator to a provided tensor.
template <AnyTensor TTensor, typename TFunction>
class Generator : public TensorOperation<typename std::remove_cvref_t<TTensor>::value_type,
                                         std::remove_cvref_t<TTensor>::rank,
                                         Generator<TTensor, TFunction>>
{
 public:
  using typename Generator::TensorOperation::value_type;
  using Generator::TensorOperation::rank;

  template <typename T, typename F>
  Generator(T&& tensor, F&& f)
    : TensorOperation<value_type, rank, Generator<TTensor, TFunction>>(*this),
      function_(std::forward<F>(f)),
      tensor_(std::forward<T>(tensor))
  {}

  /// operator()() evaluates the unary operator and returns a tensor.
  auto operator()() const
  {
    GeneratorOperator<tensor_device_t<std::remove_cvref_t<TTensor>>>(tensor_, function_);
    return tensor_;
  }

 private:
  TFunction function_;
  TTensor tensor_;
};

template <typename TOp, typename T> Generator(TOp, T&&) -> Generator<TOp, typename to_tensor<T>::type>;


/// @brief Fill fills the tensor with the provided value.
template <TensorConvertible TTensor>
void Fill(TTensor&& tensor, typename to_tensor<TTensor>::type::value_type val)
{
  return Generator(FillOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor), val);
}


/// @brief Applies the generator sequentially to the provided tensor.
///
/// Example:
///   Function(tensor, [n = 0] { return n++; });
template <TensorConvertible TTensor, typename F>
void Generate(TTensor&& tensor, F&& gen)
{
  return Generator(GeneratorOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor), std::forward<F>(gen));
}


}   // end of namespace grid

#endif  // GRID_TENSOR_GENERATOR_H
