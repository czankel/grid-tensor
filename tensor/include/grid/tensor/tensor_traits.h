//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_TRAITS_H
#define GRID_TENSOR_TENSOR_TRAITS_H

namespace grid {

template <typename _Tensor>
inline constexpr bool is_tensor_v =
  //std::is_class_v<typename std::remove_cvref_t<_Tensor>> &&
  requires (const _Tensor& t) { t.Rank(); t.Dimensions(); t.Strides(); t.Data(); };

// FIXME add more rquirements && decltype... is_tensor?
/// is_operator_v<_Operator> returns true if the template is derived from TensorOp
template <typename _Operator>
inline constexpr bool is_operator_v = requires (const _Operator& t) { t.operator()(); };

// FIXME: use decltype??
// helper class for identifying the result tensor of a tensor operation class
template <typename _Operator> struct result_of;
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _Operator,
          template <typename, size_t, typename...> typename _Tensor, size_t _Rank, typename _T, typename _Allocator, typename... _Tensors>
struct result_of<_Operator<_Tensor, _T, _Rank, _Allocator, _Tensors...>>
{
  using type = _Tensor<_T, _Rank, _Allocator>;
  // FIXME decltype(std::invoke_return<&_Operator::operator(),_Operator>);
};

// FIXME: what's this?
// helper class for getting the tensor type of the tensor or tensor operation, which returns the default tensor.
template <typename> struct to_tensor;

template <template <typename, size_t, typename...> typename _Tensor, typename _T, size_t _Rank, typename _Allocator>
struct to_tensor<_Tensor<_T, _Rank, _Allocator>>
{
  using type = _Tensor<_T, _Rank, _Allocator>;
};

template <typename _Tensor>
requires (is_operator_v<_Tensor>)
struct to_tensor<_Tensor>
{
  using type = typename result_of<_Tensor>::type;
};


// is_same_tensor_as<TENSOR1, TENSOR2> is a helper function that returns a true/false type if
// TENSOR1 is the same tensor type as TENSOR2.
// Example: is_same_tensor_as<tensor, some_device_ns::Tensor>
template <typename, template <typename, size_t, typename> typename> struct is_same_tensor_as : std::false_type {};
template <template <typename, size_t, typename> typename _Tensor, typename _Tp, size_t _Rank, typename _Allocator>
struct is_same_tensor_as<_Tensor<_Tp, _Rank, _Allocator>, _Tensor> : std::true_type {};


// tensor_is_assignable<TO,FROM> is a helper that returns a true/false type if TO = FROM is valid.
template <typename, template <typename, size_t, typename...> typename> struct tensor_is_assignable;
template <template <typename, size_t, typename...> typename _Tensor1,
          template <typename, size_t, typename...> typename _Tensor2,
          typename _Tp, size_t _Rank, typename... _Allocator>
struct tensor_is_assignable<_Tensor1<_Tp, _Rank, _Allocator...>, _Tensor2>
 : std::is_assignable<_Tensor2<_Tp, _Rank>, _Tensor1<_Tp, _Rank, _Allocator...>>
{};


template <typename _Tensor>
using tensor_data_return_type = decltype(std::declval<_Tensor>().Data());

template<class _Tensor>
struct tensor_is_primitive_helper :
std::integral_constant<bool, std::is_pointer_v<tensor_data_return_type<_Tensor>> &&
                             std::is_arithmetic_v<std::remove_pointer_t<tensor_data_return_type<_Tensor>>>> {};

template<class _Tensor>
struct tensor_is_primitive : tensor_is_primitive_helper<std::remove_reference_t<_Tensor>> {};

#if 0 // FIXME: doesn't really work...
template<typename _Functor, typename _Class>
struct is_primitive_pointer :
//std::integral_constant<bool, std::is_pointer_v<std::invoke_result_t<decltype(&_Tensor::Data),_Tensor>> &&
std::integral_constant<bool, std::is_pointer_v<std::invoke_result_t<_Functor,_Class>> &&
                             std::is_arithmetic_v<std::remove_pointer_t<std::invoke_result_t<_Functor,_Class>>>> {};
/// PrimitiveTensor requires that the buffer of the tensor is of a primitive type and can be memory copyable.
template <typename _Tensor>
  //std::remove_cv<std::remove_reference<T>::type>::type.
concept PrimitiveTensor = is_tensor_v<_Tensor> && is_primitive_pointer<decltype(&std::remove_reference_t<_Tensor>::Data), std::remove_reference_t<_Tensor>>::value;

//decltype(_Tensor::Data()) -> _Tensor::value_type; // FIXME: &&  Data() is pointer of primitive type

/// PrimitiveOperator requires that the operator can be converted to a primitive type and be memory copyable.
template <typename _Operator>
concept PrimitiveOperator = is_operator_v<_Operator>; // FIXME: ensure it's primitive..

#endif


} // end of namespace grid

#endif // GRID_TENSOR_TENSOR_TRAITS_H
