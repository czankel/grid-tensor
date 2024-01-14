//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_CONCEPTS_H
#define GRID_TENSOR_CONCEPTS_H

namespace grid {


/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename T> struct Uninitialized { using value_type = T; };


//
// Tensor Traits
//

/// is_tensor_v checks if a type is a tensor, which requires to have the member functions:
/// Dimensions, Strides, and Data.
template <typename TTensor>
inline constexpr bool is_tensor_v =
  std::is_class_v<typename std::remove_cvref_t<TTensor>> &&
  requires (const TTensor& t) { t.Rank(); t.Dimensions(); t.Strides(); t.Data(); };


/// is_operator_v<Operator> checks if a type is tensor operator, which requires to have a member
/// operator()() overload.
/// TODO: check also template signature
template <typename TOperator>
inline constexpr bool is_operator_v = requires (const TOperator& t) { t.operator()(); };

// to_tensor provides the type of the tensor or the type of the tensor resulting from a  tensor operation
template <typename> struct to_tensor;

template <typename TTensor> requires (is_tensor_v<TTensor> && !is_operator_v<TTensor>)
struct to_tensor<TTensor>
{
  using type = TTensor;
};

template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename TOperator,
          template <typename, size_t, typename...> typename TTensor,
          typename T, size_t TRank, typename... TTensors>
struct to_tensor<TOperator<TTensor, T, TRank, TTensors...>>
{
  using type = decltype(std::declval<TOperator<TTensor, T, TRank, TTensors...>>().operator()());
};


// is_same_tensor<TENSOR1, TENSOR2> checks if two tensors are of the same type.
template <typename, template <typename, size_t, typename...> typename> struct is_same_tensor_as : std::false_type {};

template <template <typename, size_t, typename...> typename TTensor, typename T, size_t TRank, typename... TAllocator>
struct is_same_tensor_as<TTensor<T, TRank, TAllocator...>, TTensor> : std::true_type {};


// tensor_is_primitive checks if a tensor includes a Data() member function thatJ returns an
// artihmetic pointer type.
template <typename TTensor> using tensor_data_return_type = decltype(std::declval<TTensor>().Data());

template<class TTensor> struct tensor_is_primitive_helper :
  std::integral_constant<bool, std::is_pointer_v<tensor_data_return_type<TTensor>> &&
                               std::is_arithmetic_v<std::remove_pointer_t<tensor_data_return_type<TTensor>>>> {};

template<class TTensor>
struct tensor_is_primitive : tensor_is_primitive_helper<std::remove_reference_t<TTensor>> {};

//
// Concepts
//

/// Arithmetic defines an arithmetic type, such as integer, float, etc.
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/// AnyTensor requires that the provided argument is a tensor.
template <typename TTensor>
concept AnyTensor = is_tensor_v<TTensor>;

/// PrimitiveTensor requires that the buffer is directly accessible and consists of primitive types,
template <typename TTensor>
concept PrimitiveTensor = AnyTensor<TTensor> && tensor_is_primitive<TTensor>::value;

/// AnyOperator requires that the provided argument is a tensor operator.
template <typename TOperator>
concept AnyOperator = is_operator_v<TOperator>;

/// TensorConvertible requires that the tensor is a tensor or functor that returns a tensor
template <typename TTensor>
concept TensorConvertible = is_tensor_v<TTensor> || is_operator_v<TTensor>;

// FIXME doesn't work for Views directly, assuming it converts View to Tensor before??
template <typename TFrom, template <typename, size_t, typename...> typename TTensor>
struct tensor_is_convertible_to : std::is_assignable<TTensor<typename TFrom::value_type, TFrom::rank>, TFrom> {};

template <typename TFrom, template <typename, size_t, typename...> typename TTensor>
concept ConvertibleTo = std::is_class_v<std::remove_cvref_t<TFrom>> && tensor_is_convertible_to<std::remove_cvref_t<TFrom>, TTensor>::value;

} // end of namespace grid

#endif // GRID_TENSOR_CONCEPTS_H