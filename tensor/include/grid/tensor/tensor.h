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

#include <grid/tensor/tensor_traits.h>

namespace grid {


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


// Concepts

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


/// TensorBase provides a base class for derived "device" tensor implementations with
/// optimizations specific to CPUs and accelerators.
struct TensorBase
{
  friend std::ostream& operator<<(std::ostream& os, const AnyTensor auto& tensor);
};

// Tensor basic arithmetic operations

template <template <typename, size_t, auto...> typename, typename, size_t, typename... > struct TensorAdd;


// Operator overloading

// operator+ (TensorType, TensorType)
template <AnyConvertibleTensor _Tensor1, AnyConvertibleTensor _Tensor2>
auto operator+(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  return TensorAdd(std::forward<_Tensor1>(tensor1), std::forward<_Tensor2>(tensor2));
}


// helper function to extra brace-initializer list
template <typename _T, size_t _Count>
inline constexpr std::array<_T, _Count>
get_array(std::initializer_list<_T>&& init)
{
  std::array<_T, _Count> arr;
  std::copy(init.begin(), init.end(), arr.begin());
  return arr;
}

// helper function to initialize the std:array from an initializer list
template <typename _T, size_t _M, size_t _N>
inline constexpr std::array<_T, _M * _N>
get_array(std::initializer_list<std::initializer_list<_T>>&& init)
{
  std::array<_T, _M * _N> arr{};
  auto line_it = arr.begin();
  for (auto it : init)
  {
    std::copy(it.begin(), it.end(), line_it);
    line_it += _N;
  }
  return arr;
}

template <typename _T, size_t _N>
inline constexpr std::array<_T, _N>
get_array(const _T(&init)[_N])
{
  std::array<_T, _N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

template <typename _T, size_t _N>
inline constexpr std::array<_T, _N>
get_array(_T(&&init)[_N])
{
  std::array<_T, _N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// helper function to return the strides from dimensions. Use: make_strides<TYPE>(std::array)
template <typename _T, size_t _Rank, size_t... Is>
constexpr std::array<ssize_t, _Rank>
make_strides_impl(const std::array<size_t, _Rank>& dims, std::index_sequence<Is...>)
{
  auto multiply = [&dims](size_t index) {
    ssize_t res = sizeof(_T);
    for (size_t i = 0; i < _Rank - 1 - index; i++)
      res *= dims[_Rank - 1 - i];
    return res;
  };
  return std::array<ssize_t, _Rank>{multiply(Is)...};
}

template <typename _T, size_t _Rank, typename Indices = std::make_index_sequence<_Rank>>
std::array<ssize_t, _Rank> make_strides(const std::array<size_t, _Rank>& dims)
{
  return make_strides_impl<_T>(dims, Indices{});
}

/// operator<< outputs the tensor buffer.
std::ostream& operator<<(std::ostream& os, const grid::AnyTensor auto& tensor)
{
  using value_type = std::remove_reference_t<decltype(tensor)>::value_type;
  constexpr size_t rank = tensor.Rank();

  auto dims = tensor.Dims();
  auto strides = tensor.Strides();

  std::function<void(int, const value_type*&)> print;
  print = [&os, &tensor, &dims, &strides, &print](size_t index, const value_type*& ptr) {
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
  print(0, ptr);
  os << std::flush;

  return os;
}

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_H
