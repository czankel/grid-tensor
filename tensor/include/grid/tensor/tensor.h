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
#include <numeric>

#include <grid/tensor/tensor_traits.h>

// FIXME
/// Tensor describes ... ///  implementation of "AI Tensors".
///These tensors implmeneted t storing data and typical vector and matrix operations rather than a mathematical or
// physical definition of a Tensor.
///
/// Tensor Operations  provides a base class for identifying tensor operator classes.
///
///  typedef value_type;
///

namespace grid {

constexpr static size_t kMaxRank = 4;

/// MemoryMapped is used as a non-type template parameter declaring a memory-mapped tensor type.
/// Defining a memory-mapped tensor ... TensorImplementation<double, 2, grid::MemoryMapped{})
struct MemoryMapped {};

/// TensorView is used as a non-type template parameter declaring a tensor view type.
/// This is only used internally.
struct TensorView {};

/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename T> struct Uninitialized { using type = T; };


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
template <typename _Tensor, template <typename, size_t, auto...> typename _DeviceTensor>
concept ConvertibleTensorFor = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && is_same_device_v<_Tensor, _DeviceTensor>;

/// AnyConvertibleTensor requires that the provided argument can be converted to a Tensor.
template <typename _Tensor>
concept AnyConvertibleTensor = is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>;

/// TensorRank<RANK> requires that the provided argument is a tensor of the rank RANK.
template <typename _Tensor, size_t _Rank>
concept TensorRank = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && _Tensor::Rank() == _Rank;

/// TensorNotRank<RANK> requires that the provided argument is a tensor that is not of the rank RANK.
template <typename _Tensor, size_t _Rank>
concept TensorNotRank = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && _Tensor::Rank() != _Rank;

// Viewable requires that a tensor can be made "viewable", which means it is not a view itself.
template <typename _Tensor>
concept Viewable = requires (const _Tensor& t) { t.View; };

/// TensorViewFor<DEVICE> requires that the provided argument is a tensor view and for a specific device
template <typename _Tensor, template <typename, size_t, auto...> typename _DeviceTensor>
concept TensorViewFor = !Viewable<_Tensor> && is_same_device_v<_Tensor, _DeviceTensor>;


// Supported basic arithmetic operations for all Tensor implementations.

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


// helper function to get an array from a brace-initializer list.
template <typename _T, size_t... _Ns>
inline constexpr std::array<_T, sizeof...(_Ns)>
get_array_impl(std::initializer_list<_T>&& init, std::index_sequence<_Ns...>)
{
  return std::array<_T, sizeof...(_Ns)>{ *(init.begin() + _Ns) ... };
}

template <typename _T, size_t _N, typename _Ns = std::make_index_sequence<_N>>
inline constexpr std::array<_T, _N>
get_array(std::initializer_list<_T>&& init)
{
  return get_array_impl(std::move(init), _Ns{});
}

// helper function to return an array from a two-dimensional initializer list
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

// helper function to return an array from a three-dimensional initializer list
template <typename _T, size_t _C, size_t _M, size_t _N>
inline constexpr std::array<_T, _C * _M * _N>
get_array(std::initializer_list<std::initializer_list<std::initializer_list<_T>>>&& init)
{
  printf("C\n");
  std::array<_T, _C * _M * _N> arr{};
  auto line_it = arr.begin();
  for (auto lt : init)
  {
    for (auto it : lt)
    {
      std::copy(it.begin(), it.end(), line_it);
      line_it += _N;
    }
  }
  return arr;
}

// helper function to re turn an array from a c-array.
template <typename _T, size_t _N>
inline constexpr std::array<_T, _N>
get_array(const _T(&init)[_N])
{
  printf("D\n");
  std::array<_T, _N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// helper function to re turn an array from an rvalue c-array.
template <typename _T, size_t _N>
inline constexpr std::array<_T, _N>
get_array(_T(&&init)[_N])
{
  printf("E\n");
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
  using value_type = typename std::remove_reference_t<decltype(tensor)>::value_type;
  size_t rank = tensor.Rank();

  auto dims = tensor.Dims();
  auto strides = tensor.Strides();

  std::function<void(int, const value_type*&)> print;
  print = [&os, &dims, &strides, &print, &rank](size_t index, const value_type* ptr) {
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
