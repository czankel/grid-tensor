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
struct TensorBaseOp {};

// FIXME add more rquirements
/// is_tensor_op_v<_TensorOp> returns true if the template is derived from TensorOp
template <typename _TensorOp>
inline constexpr bool is_tensor_op_v = requires (const _TensorOp& t) { t.operator()(); };


//std::is_base_of_v<TensorBaseOp, std::remove_cvref_t<_TensorOp>>;

template <typename _Tensor>
inline constexpr bool is_tensor_v =
  std::is_class_v<typename std::remove_cvref_t<_Tensor>> &&
  requires (const _Tensor& t) { t.Rank(), t.Dimensions(); t.Strides(); };

// helper functions to identify if a Tensor or TensorOp is for a specific device
namespace details
{
  template <template <typename, size_t, typename...> typename _Tensor,
            typename _T, size_t _Rank, typename _Allocator>
  std::true_type test_ptr_conv(const volatile _Tensor<_T, _Rank, _Allocator>*);
  template<typename, typename>
  std::false_type test_ptr_conv(const volatile void*);

  template <typename _Tensor, template <typename, size_t, typename...> typename _DeviceTensor>
  auto test_is_same_device(int)
    -> decltype(test_ptr_conv<_DeviceTensor>(static_cast<_Tensor*>(nullptr)));

  template<typename, typename>
  auto test_is_same_device(...) -> std::true_type; // private or ambiguous base
}

template <typename _Tensor, template <typename, size_t, typename...> typename _DeviceTensor>
struct is_same_device :
    std::integral_constant<
        bool,
        std::is_class<_Tensor>::value &&
        decltype(details::test_is_same_device<_Tensor, _DeviceTensor>(0))::value
    > {};

// TODO: these are not really c FIXME

template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, typename...> typename _Tensor, size_t _Rank, typename _T, typename... _Tensors>
struct is_same_device<_TensorOp<_Tensor, _T, _Rank, _Tensors...>, _Tensor> : std::true_type {};


template <typename _Tensor, template <typename, size_t, typename...> typename _DeviceTensor>
inline constexpr bool is_same_device_v = is_same_device<std::remove_cvref_t<_Tensor>, _DeviceTensor>::value;


// helper class to check if the variadic tensor parameter arguments include a specific non-type
#if 0
template <typename, auto, typename = void> struct has_any : std::false_type {};
template <template <typename, size_t, typename...> typename _Tensor, typename _T, size_t _Rank, typename... _Allocator, auto _Allocator>
struct has_any<_Tensor<_T, _Rank, _Allocator, _Args...>, _Arg, typename std::enable_if<((_Arg == _Args) || ...)>::type> : std::true_type {};
#endif
// helper class for identifying the result tensor of a tensor operation class
template <typename _TensorOp> struct result_of;
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
          template <typename, size_t, typename...> typename _Tensor, size_t _Rank, typename _T, typename _Allocator, typename... _Tensors>
struct result_of<_TensorOp<_Tensor, _T, _Rank, _Allocator, _Tensors...>>
{
  using type = _Tensor<_T, _Rank, _Allocator>;
};

// helper class for getting the tensor type of the tensor or tensor operation, which returns the default tensor.
template <typename> struct to_tensor;

template <template <typename, size_t, typename...> typename _Tensor, typename _T, size_t _Rank, typename _Allocator>
struct to_tensor<_Tensor<_T, _Rank, _Allocator>>
{
  using type = _Tensor<_T, _Rank, _Allocator>;
};

template <typename _Tensor>
requires (is_tensor_op_v<_Tensor>)
struct to_tensor<_Tensor>
{
  using type = typename result_of<_Tensor>::type;
};


} // end of namespace grid

#endif // GRID_TENSOR_TENSOR_TRAITS_H
