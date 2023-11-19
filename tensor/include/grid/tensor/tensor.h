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

/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename> struct Uninitialized {};

//
// Tensor Traits
//

/// is_tensor_v checks if a type is a tensor, which requires to have the member functions:
/// Dimensions, Strides, and Data.
template <typename _Tensor>
inline constexpr bool is_tensor_v =
  std::is_class_v<typename std::remove_cvref_t<_Tensor>> &&
  requires (const _Tensor& t) { t.Rank(); t.Dims(); t.Strides(); t.Data(); };


/// is_operator_v<_Operator> checks if a type is tensor operator, which requires to have a member
/// operator()() overload.
/// TODO: check also template signature
template <typename _Operator>
inline constexpr bool is_operator_v = requires (const _Operator& t) { t.operator()(); };


// to_tensor provides the type of the tensor or the type of the tensor resulting from a  tensor operation
template <typename> struct to_tensor;

template <typename _Tensor> requires (is_tensor_v<_Tensor> && !is_operator_v<_Tensor>)
struct to_tensor<_Tensor>
{
  using type = _Tensor;
};

template <typename _Operator> requires is_operator_v<_Operator>
struct to_tensor<_Operator>
{
  using type = typename std::invoke_result<decltype(&_Operator::operator()), _Operator>::type;
};


// is_same_tensor<TENSOR1, TENSOR2> checks if two tensors are of the same type.
template <typename, template <typename, size_t, auto...> typename> struct is_same_tensor_as : std::false_type {};

template <template <typename, size_t, auto...> typename _Tensor, typename _Tp, size_t _Rank, auto... _Args>
struct is_same_tensor_as<_Tensor<_Tp, _Rank, _Args...>, _Tensor> : std::true_type {};


// tensor_is_convertible_to<FROM,TO> check if TO = FROM is a valid assignment.
template <typename, template <typename, size_t, auto...> typename> struct tensor_is_convertible_to;

template <template <typename, size_t, auto...> typename _Tensor1,
          template <typename, size_t, auto...> typename _Tensor2,
          typename _Tp, size_t _Rank, auto... _Args>
struct tensor_is_convertible_to<_Tensor1<_Tp, _Rank, _Args...>, _Tensor2>
 : std::is_assignable<_Tensor2<_Tp, _Rank>, _Tensor1<_Tp, _Rank, _Args...>>
{};

template <template <template <typename, size_t, auto...> typename, typename, size_t, typename...> typename _Operator,
          template <typename, size_t, auto...> typename _Tensor,
          typename _Tp, size_t _Rank, typename... _Tensors>
struct tensor_is_convertible_to<_Operator<_Tensor, _Tp, _Rank, _Tensors...>, _Tensor>
 : std::is_assignable<_Tensor<_Tp, _Rank>, _Operator<_Tensor, _Tp, _Rank, _Tensors...>>
{};


// tensor_is_primitive checks if a tensor includes a Data() member function thatJ returns an
// artihmetic pointer type.
template <typename _Tensor> using tensor_data_return_type = decltype(std::declval<_Tensor>().Data());

template<class _Tensor> struct tensor_is_primitive_helper :
  std::integral_constant<bool, std::is_pointer_v<tensor_data_return_type<_Tensor>> &&
                               std::is_arithmetic_v<std::remove_pointer_t<tensor_data_return_type<_Tensor>>>> {};

template<class _Tensor>
struct tensor_is_primitive : tensor_is_primitive_helper<std::remove_reference_t<_Tensor>> {};

//
// Concepts
//

/// AnyTensor requires that the provided argument is a tensor.
template <typename _Tensor>
concept AnyTensor = is_tensor_v<_Tensor>;

/// PrimitiveTensor requires that the buffer is directly accessible and consists of primitive types,
template <typename _Tensor>
concept PrimitiveTensor = AnyTensor<_Tensor> && tensor_is_primitive<_Tensor>::value;

/// AnyOperator requires that the provided argument is a tensor operator.
template <typename _Operator>
concept AnyOperator = is_operator_v<_Operator>;

/// TensorConvertible requires that the tensor is a tensor or functor that returns a tensor
template <typename _Tensor>
concept TensorConvertible = is_tensor_v<_Tensor> || is_operator_v<_Tensor>;

template <typename T1, template <typename, size_t, auto...> typename T2>
concept ConvertibleTo = tensor_is_convertible_to<std::remove_cvref_t<T1>, T2>::value;

//
// Tensor basic arithmetic operations
//

template <template <typename, size_t, auto...> typename, typename, size_t, typename... > struct TensorAdd;
template <template <typename, size_t, auto...> typename, typename, size_t, typename... > struct TensorMul;
template <template <typename, size_t, auto...> typename, typename, size_t, typename... > struct TensorRmsNorm;

//
// Operator overloading
//

// operator+ (TensorType, TensorType)
template <TensorConvertible _Tensor1, TensorConvertible _Tensor2>
auto operator+(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  return TensorAdd(std::forward<_Tensor1>(tensor1), std::forward<_Tensor2>(tensor2));
}

// operator* (TensorType, TensorType)
template <TensorConvertible _Tensor1, TensorConvertible _Tensor2>
auto operator*(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  return TensorMul(std::forward<_Tensor1>(tensor1), std::forward<_Tensor2>(tensor2));
}

} // end of namespace grid

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

#endif  // GRID_TENSOR_TENSOR_H
