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

struct TensorBase;
struct TensorBaseOp;

/// is_tensor_op_v<_TensorOp> returns true if the template is derived from TensorOp
template <typename _TensorOp>
inline constexpr bool is_tensor_op_v = std::is_base_of_v<TensorBaseOp, std::remove_cvref_t<_TensorOp>>;

// is_tensor_v returns true if the type is a tensor (derived from TensorBase)
template <typename _Tensor>
inline constexpr bool is_tensor_v = std::is_base_of_v<TensorBase, std::remove_cvref_t<_Tensor>>;

// helper functions to identify if a Tensor or TensorOp is for a specific device
template <typename, template <typename, size_t, auto...> typename>
struct is_same_device : std::false_type {};

template <template <typename, size_t, auto...> typename _Tensor, typename _T, size_t _Rank, auto... _Args>
struct is_same_device<_Tensor<_T, _Rank, _Args...>, _Tensor> : std::true_type {};

template <template <template <typename, size_t, auto...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, auto...> typename _Tensor, size_t _Rank, typename _T, typename... _Tensors>
struct is_same_device<_TensorOp<_Tensor, _T, _Rank, _Tensors...>, _Tensor> : std::true_type {};

template <typename _Tensor, template <typename, size_t, auto...> typename _DeviceTensor>
inline constexpr bool is_same_device_v = is_same_device<std::remove_cvref_t<_Tensor>, _DeviceTensor>::value;

} // end of namespace grid

#endif // GRID_TENSOR_TENSOR_TRAITS_H
