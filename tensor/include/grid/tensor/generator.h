//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_GENERATOR_H
#define GRID_TENSOR_GENERATOR_H

#include "tensor.h"
#include "tensor_parameters.h"

namespace grid {

//template <template <typename> typename, typename> class GeneratorOperation;
template <typename> class GeneratorOperation;

/// @brief Generator generates a sequence of elements filling a tensor
///
/// Examples:
///   auto tensor = Generate<TENSOR>({400, 600}, [](size_t y, size_t x) -> T { ... });
///   auto tensor = Random<TENSOR, float>({200, 200});
///
// TODO: add variadic arguments passed to the generator function
template <AnyTensor TTensor, typename TOperation>
class Generator : public TensorOperation<typename std::remove_cvref_t<TTensor>::value_type,
                                         std::remove_cvref_t<TTensor>::rank,
                                         Generator<TTensor, TOperation>>
{
 public:
  using typename Generator::TensorOperation::value_type;
  using Generator::TensorOperation::rank;
  using device_type = tensor_device_t<TTensor>;

  explicit Generator(const size_t(&&dims)[rank])
    : TensorOperation<value_type, rank, Generator<TTensor, TOperation>>(*this),
      dimensions_(get_array<size_t, rank>(dims))
  {}

  Generator() = delete;
  Generator(const Generator& other) = delete;
  Generator& operator=(const Generator& other) = delete;


  /// operator()() evaluates the unary operator and returns a tensor.
  auto operator()() const
  {
    auto result = TTensor(dimensions_, Uninitialized<value_type>());
    GeneratorOperation<device_type>{}(result, operator_);
    return result;
  }

 private:
  static TOperation         operator_;
  std::array<size_t, rank>  dimensions_;
};

//
// Generator Functions
//

#if 0 // TODO: requires variadic arguments for Generators
/// @brief Fill returns a tensor filled with the provided value.
template <typename> struct FillFunction;
template <template <typename, size_t, typename...> typename TTensor, typename T, size_t N>
auto Fill(const size_t(&&dims)[N], T val)
{
  using tensor_type = decltype(TTensor(std::move(dims), Uninitialized<T>()));
  using device_type = tensor_device_t<tensor_type>;
  return Generator<tensor_type, FillFunction<device_type>>(std::move(dims));
  return Generator(FillOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor), val);
}
#endif

/// @brief Random returns a tensor of the specified dimensions filled with random values.
template <typename> struct RandomFunction;
template <template <typename, size_t, typename...> typename TTensor, typename T, size_t N>
auto Random(const size_t(&&dims)[N])
{
  using tensor_type = decltype(TTensor(std::move(dims), Uninitialized<T>()));
  using device_type = tensor_device_t<tensor_type>;
  return Generator<tensor_type, RandomFunction<device_type>>(std::move(dims));
}

}   // end of namespace grid

#endif  // GRID_TENSOR_GENERATOR_H
