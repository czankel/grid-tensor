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

namespace grid {

constexpr static size_t kMaxRank = 4;

/// ConstantAllocator is a special "allocator for constant static data.
template <size_t...> struct StaticAllocator {};

/// Broadcast defines to set the dimension to 1 ("broadcastable") in the axes argument of Tensor::View.
inline constexpr ssize_t Broadcast = -1;

/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename T> struct Uninitialized { using type = T; };

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
concept PrimitiveOperator = is_tensor_op_v<_Operator>; // FIXME: ensure it's primitive..

#endif

// Concepts

/// AnyTensor requires that the provided argument is a tensor.
template <typename _Tensor>
concept AnyTensor = is_tensor_v<_Tensor>;

template <typename _Tensor>
concept TriviallyCopyableTensor = is_tensor_v<_Tensor>;

// TODO: must be 'movable', i.e
template <typename _Tensor>
concept TriviallyMovableTensor = is_tensor_v<_Tensor> &&   std::assignable_from<_Tensor&, _Tensor>;
//std::movable<_Tensor>;

#if 0
/// TensorOperatorFor<DEVICE> requires that the provided argument is a TensorOperator for the specific DEVICE.
template <typename _TensorOperator, template <typename, size_t, typename> typename _DeviceTensor>
concept TensorOperatorFor = is_tensor_op_v<_TensorOperator> && is_same_device_v<_TensorOperator, _DeviceTensor>;
#endif


/// AnyTensorOperator requires that the provided argument is a TensorOperator
template <typename _TensorOperator>
concept AnyTensorOperator = is_tensor_op_v<_TensorOperator>;

/// ConvertibleTo<TENSOR> requires that the provided type can be converted to the specific tensor type.
template <typename _Tensor, template <typename, size_t, typename> typename _DeviceTensor>
concept ConvertibleTo = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && is_same_device_v<_Tensor, _DeviceTensor>;

// FIXME
template <typename _Tensor>
concept AnyOperator = is_tensor_op_v<_Tensor>;



/// AnyConvertibleTensor describes a type that can be converted to a tensor.
template <typename _Tensor>
concept AnyConvertibleTensor = is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>;

/// TensorRank<RANK> requires that the provided argument is a tensor of the rank RANK.
template <typename _Tensor, size_t _Rank>
concept TensorRank = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && _Tensor::rank == _Rank;

#if 1 // FIXME: feels odd to require?
/// TensorNotRank<RANK> requires that the provided argument is a tensor that is not of the rank RANK.
template <typename _Tensor, size_t _Rank>
concept TensorNotRank = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && _Tensor::rank != _Rank;
#endif

// Viewable requires that a tensor can be made "viewable" and provides a "View" method.
// FIXME: ensure it's a tensor...
template <typename _Tensor>
concept Viewable = requires (const _Tensor& t) { t.View; };

#if 0
/// TensorViewFor<DEVICE> requires that the provided argument is a tensor view and for a specific device
template <typename _Tensor, template <typename, size_t, typename> typename _DeviceTensor>
concept TensorViewFor = Viewable<_Tensor> && is_same_device_v<_Tensor, _DeviceTensor>;
#endif

// Empty arithmetic operator declaraions that can be specialized for different tensor types.
// Specializing tensors enables also the use of the acellerators

template <template <typename, size_t, typename> typename, typename, size_t, typename... > struct TensorAdd;
template <template <typename, size_t, typename> typename, typename, size_t, typename... > struct TensorMul;
template <template <typename, size_t, typename> typename, typename, size_t, typename... > struct TensorRmsNorm;


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
  std::array<_T, _N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// helper function to re turn an array from an rvalue c-array.
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

} // end of namespace grid

// FIXME: move to output.h?
/// operator<< outputs the tensor buffer.
std::ostream& operator<<(std::ostream& os, const grid::AnyTensor auto& tensor)
{
  using value_type = typename std::remove_reference_t<decltype(tensor)>::value_type;
  size_t rank = std::decay_t<decltype(tensor)>::rank;

  auto dims = tensor.Dimensions();
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


#endif  // GRID_TENSOR_TENSOR_H
