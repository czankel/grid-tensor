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
namespace details
{
  template <template <typename, size_t, auto...> typename _Tensor,
            typename _T, size_t _Rank, auto... _Args>
  std::true_type test_ptr_conv(const volatile _Tensor<_T, _Rank, _Args...>*);
  template<typename, typename>
  std::false_type test_ptr_conv(const volatile void*);

  template <typename _Tensor, template <typename, size_t, auto...> typename _DeviceTensor>
  auto test_is_same_device(int)
    -> decltype(test_ptr_conv<_DeviceTensor>(static_cast<_Tensor*>(nullptr)));

  template<typename, typename>
  auto test_is_same_device(...) -> std::true_type; // private or ambiguous base
}

template <typename _Tensor, template <typename, size_t, auto...> typename _DeviceTensor>
struct is_same_device :
    std::integral_constant<
        bool,
        std::is_class<_Tensor>::value &&
        decltype(details::test_is_same_device<_Tensor, _DeviceTensor>(0))::value
    > {};

template <template <template <typename, size_t, auto...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, auto...> typename _Tensor, size_t _Rank, typename _T, typename... _Tensors>
struct is_same_device<_TensorOp<_Tensor, _T, _Rank, _Tensors...>, _Tensor> : std::true_type {};

template <typename _Tensor, template <typename, size_t, auto...> typename _DeviceTensor>
inline constexpr bool is_same_device_v = is_same_device<std::remove_cvref_t<_Tensor>, _DeviceTensor>::value;

} // end of namespace grid

#endif // GRID_TENSOR_TENSOR_TRAITS_H
