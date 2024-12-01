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

template <template <typename> typename, typename> class GeneratorOperator;

template <typename> struct FillOperator;
template <typename> struct FunctionOperator;

/// @brief Generator applies the given generator operator to a provided tensor.
template <typename TOperator, AnyTensor TTensor, typename TFunction>
class Generator : public TensorOperator<typename std::remove_cvref_t<TTensor>::value_type,
                                        std::remove_cvref_t<TTensor>::rank,
                                        Generator<TOperator, TTensor, TFunction>>
{
 public:
  using typename Generator::TensorOperator::value_type;
  using Generator::TensorOperator::rank;

  template <typename T, typename F>
  Generator(TOperator, T&& tensor, F&& f)
    : TensorOperator<value_type, rank, Generator<TOperator, TTensor, TFunction>>(*this),
      function_(std::forward<F>(f)),
      tensor_(std::forward<T>(tensor))
  {}

  /// operator()() evaluates the unary operator and returns a tensor.
  auto operator()() const
  {
    //GeneratorOperator<TOperator, tensor_device_t<std::remove_cvref_t<TTensor>>>(tensor_, function_);
    operator_(tensor_, function_);
/*
    using ResultTensor = Tensor<value_type, rank, DeviceMemory<tensor_device_t<TTensor>>>;
    auto result = ResultTensor(tensor_.Dimensions(), Uninitialized<value_type>{});
    operator_(tensor_, result, function_);
    */
    return tensor_;
  }

 private:
  static TOperator operator_;
  TFunction function_;
  TTensor tensor_;
};

template <typename TOp, typename T, typename F> Generator(TOp, T&&, F&&)
  -> Generator<TOp, typename to_tensor<T>::type, F>;


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
  return Generator(FunctionOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor), std::forward<F>(gen));
}


}   // end of namespace grid

#endif  // GRID_TENSOR_GENERATOR_H
