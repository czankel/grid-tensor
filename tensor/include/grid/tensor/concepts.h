//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_CONCEPTS_H
#define GRID_TENSOR_CONCEPTS_H

#include "tensor_traits.h"

namespace grid {

/// AnyTensor requires that the provided argument is a tensor.
template <typename _Tensor>
concept AnyTensor = is_tensor_v<_Tensor>;

/// PrimitiveType requires that the buffer is directly accessible and consists of primitive types,
template <typename _Tensor>
concept PrimitiveTensor = AnyTensor<_Tensor> && tensor_is_primitive<_Tensor>::value;

//std::is_same_v<typename std::remove_cvref_t<_Tensor>::pointer,
 //       std::remove_const_t<std::invoke_result_t<decltype(&std::remove_reference_t<_Tensor>::Data),
  //      std::remove_reference_t<_Tensor>>>>;

// TODO: TriviallyMappable requires that the tensor can be mapped to a trivial buffer but may incur a cost.

/// AnyOperator requires that the provided argument is a tensor operator.
template <typename _Operator>
concept AnyOperator = is_operator_v<_Operator>;

// FIXME: ConvertibleTensor??
/// TensorConvertible requires that the tensor is a tensor or functor that returns a tensor
template <typename _Tensor>
concept TensorConvertible = is_tensor_v<_Tensor> || is_operator_v<_Tensor>;

//template <typename _Tp, typename _Tensor>
//concept ConvertibleTo = std::assignable_from<_Tp&, _Tp&>; // FIXME || opertor()...

#if 0
template <typename, typename> struct tensor_is_assignable_helper;
template <template <typename, size_t, typename> typename _Tensor, typename _T, size_t _Rank, typename _Allocator>
struct tensor_is_assignable_helper<_Tensor<_T, _Rank, _Allocator>, _Tensor> : std::true_type {};
#endif

template <typename T1, template <typename, size_t, typename...> typename T2>
concept ConvertibleTo = tensor_is_assignable<std::remove_cvref_t<T1>, T2>::value;



// && is_tensor_v<tensor_helper<_To>::type>;
///_To<_To::value_typer>>; //&& std::assignable_from<_Tensor1&, _Tensor2<typename _Tensor2::value_type, _Tensor2::Rank&, _Tensor2::allocator_type>; // FIXME || opertor()...

//std::is_same_v<_Tp, _Tensor>; /// FIXME


/////////////////////////////////
// FIXME: remove and use something different? I.e. _Tensor::rank directly?
#if 0
/// TensorRank<RANK> requires that the provided argument is a tensor of the rank RANK.
/// FIXME: include operator here??
template <typename _Tensor, size_t _Rank>
concept TensorRank = (is_tensor_v<_Tensor> || is_operator_v<_Tensor>) && _Tensor::rank == _Rank;

/// TensorNotRank<RANK> requires that the provided argument is a tensor that is not of the rank RANK.
template <typename _Tensor, size_t _Rank>
concept TensorNotRank = (is_tensor_v<_Tensor> || is_operator_v<_Tensor>) && _Tensor::rank != _Rank;
#endif
#if 0 // FIXME: future
/// TriviallyMapableTensor requires that the tensor can be trivially accesser or mapped to memory.
template <typename _Tensor>
concept TriviallyMappable = ... // FIXME
#endif
#if 0
/// ConvertibleTo requires that the tensor is convertible to a specific tensor.
/// This is typically for "trivial" and "device" tensors.
template <typename _Tensor, typename _Tensor>
concept ConvertibleTo = is_convertible_to<_Tensor, _Tensor>; // FIXME
/// ConvertibleTo<TENSOR> requires that the provided tensor can be converted to the specific tensor type.
template <typename _Tensor, template <typename, size_t, typename> typename _DeviceTensor>
#endif



} // end of namespace grid

#endif  // GRID_CONCEPTS_TENSOR_H
