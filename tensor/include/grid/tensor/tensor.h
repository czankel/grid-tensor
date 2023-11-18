//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_H
#define GRID_TENSOR_TENSOR_H

#include <concepts>
#include <functional>
#include <iomanip>
#include <iostream>

namespace grid {

/// Non-type enumerator for tensor template parameters
enum TensorType
{
  kMemoryMapped,      /// Memory mapped tensor
};


/// TensorBaseOp provides a base class for identifying tensor operator classes.
///
/// All Tensor operators/operations are required to derive from this class.
/// They must include a type for the resulting tensor and one for the underlying
/// primitive type:
///
///  typedef tensor_type;
///  typedef value_type;
///
/// Tensor operators must also provide the following functions, similar to Tensors.
///
///  size_t Rank()
struct TensorBaseOp {};


/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename> struct Uninitialized {};

struct TensorBase;
struct TensorBaseOp;

//
// Tensor Traits
//

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

//
// Concepts
//

/// TensorFor<DEVICE> requires that the provided argument is a Tensor for the specific DEVICE.
template <typename _Tensor, template <typename, size_t, auto...> typename _DeviceTensor>
concept TensorFor = is_tensor_v<_Tensor> && is_same_device_v<_Tensor, _DeviceTensor>;

/// AnyTensor requires that the provided argument is a Tensor
template <typename _Tensor>
concept AnyTensor = is_tensor_v<_Tensor>;

/// TensorOpFor<DEVICE> requires that the provided argument is a TensorOp for the specific DEVICE.
template <typename _TensorOp, template <typename, size_t, auto...> typename _DeviceTensor>
concept TensorOpFor = is_tensor_op_v<_TensorOp> && is_same_device_v<_TensorOp, _DeviceTensor>;

/// AnyTensorOp requires that the provided argument is a TensorOp
template <typename _TensorOp>
concept AnyTensorOp = is_tensor_op_v<_TensorOp>;

/// ConvertibleTensorFor<DEVICE> requires that the provided argument can be converted to a Tensor
/// for the specified DEVICE. Currently, these are Tensors and TensorOps.
template <typename _Tensor, template <typename, size_t, auto...> typename _DeviceTensor>
concept ConvertibleTensorFor = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && is_same_device_v<_Tensor, _DeviceTensor>;

/// AnyConvertibleTensor requires that the provided argument can be converted to a Tensor.
template <typename _Tensor>
concept AnyConvertibleTensor = is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>;

/// TensorRank<RANK> requires that the provided argument is a tensor of the rank RANK.
template <typename _Tensor, size_t _Rank>
concept TensorRank = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && _Tensor::Rank() == _Rank;

template <typename _Tensor, size_t _Rank>
concept TensorNotRank = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && _Tensor::Rank() != _Rank;

/// TensorBase provides a base class for derived "device" tensor implementations with
/// optimizations specific to CPUs and accelerators.
struct TensorBase
{
  friend std::ostream& operator<<(std::ostream& os, const AnyTensor auto& tensor);
};

// Tensor basic arithmetic operations

template <template <typename, size_t, auto...> typename, typename, size_t, typename... > struct TensorAdd;
template <template <typename, size_t, auto...> typename, typename, size_t, typename... > struct TensorMul;
template <template <typename, size_t, auto...> typename, typename, size_t, typename... > struct TensorRmsNorm;


// Operator overloading

// operator+ (TensorType, TensorType)
template <AnyConvertibleTensor _Tensor1, AnyConvertibleTensor _Tensor2>
auto operator+(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  return TensorAdd(std::forward<_Tensor1>(tensor1), std::forward<_Tensor2>(tensor2));
}

// operator* (TensorType, TensorType)
template <AnyConvertibleTensor _Tensor1, AnyConvertibleTensor _Tensor2>
auto operator*(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  return TensorMul(std::forward<_Tensor1>(tensor1), std::forward<_Tensor2>(tensor2));
}


/// operator<< outputs the tensor buffer.
std::ostream& operator<<(std::ostream& os, const grid::AnyTensor auto& tensor)
{
  using value_type = typename std::remove_reference_t<decltype(tensor)>::value_type;
  size_t rank = tensor.Rank();

  auto dims = tensor.Dims();
  auto strides = tensor.Strides();

  std::function<void(int, const value_type*&)> print;
  print = [&os, &dims, &strides, &print, &rank](size_t index, const value_type*& ptr) {
    os << "{ ";
    if (index < rank -1)
    {
      for (size_t i = dims[index]; i > 0; i--)
      {
        print(index + 1, ptr);
        if (i != 1)
          os << ", ";
        else
          os << " }";
        ptr += strides[index] / sizeof(*ptr);
      }
    }
    else
    {
      auto* p = ptr;
      for (size_t i = dims[index]; i > 0; i--)
      {
        os << *p;
        if (i != 1)
          os << ", ";
        else
          os << " }";
        p += strides[index] / sizeof(*ptr);
      }
    }
  };
  const value_type* ptr = reinterpret_cast<const value_type*>(tensor.Data());
  if (rank > 0)
    print(0, ptr);
  else
    os << "{ " << *ptr << " }";

  os << std::flush;

  return os;
}

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_H
