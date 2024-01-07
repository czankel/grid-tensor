//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_H
#define GRID_TENSOR_TENSOR_H

#include <functional>
#include <iostream>

namespace grid {

/// StaticAllocator is a special "allocator" for constant static data.
template <size_t...> struct StaticAllocator {};

/// NoAllocator is a spcial "allocator" for an externally managed buffer.
struct NoAllocator {};


/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename T> struct Uninitialized { using value_type = T; };

//
// Tensor Traits
//

/// is_tensor_v checks if a type is a tensor, which requires to have the member functions:
/// Dimensions, Strides, and Data.
template <typename TTensor>
inline constexpr bool is_tensor_v =
  std::is_class_v<typename std::remove_cvref_t<TTensor>> &&
  requires (const TTensor& t) { t.Rank(); t.Dimensions(); t.Strides(); t.Data(); };


/// is_operator_v<Operator> checks if a type is tensor operator, which requires to have a member
/// operator()() overload.
/// TODO: check also template signature
template <typename TOperator>
inline constexpr bool is_operator_v = requires (const TOperator& t) { t.operator()(); };

// to_tensor provides the type of the tensor or the type of the tensor resulting from a  tensor operation
template <typename> struct to_tensor;

template <typename TTensor> requires (is_tensor_v<TTensor> && !is_operator_v<TTensor>)
struct to_tensor<TTensor>
{
  using type = TTensor;
};

template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename TOperator,
          template <typename, size_t, typename...> typename TTensor,
          typename T, size_t TRank, typename... TTensors>
struct to_tensor<TOperator<TTensor, T, TRank, TTensors...>>
{
  using type = decltype(std::declval<TOperator<TTensor, T, TRank, TTensors...>>().operator()());
};


// is_same_tensor<TENSOR1, TENSOR2> checks if two tensors are of the same type.
template <typename, template <typename, size_t, typename...> typename> struct is_same_tensor_as : std::false_type {};

template <template <typename, size_t, typename...> typename TTensor, typename T, size_t TRank, typename... TAllocator>
struct is_same_tensor_as<TTensor<T, TRank, TAllocator...>, TTensor> : std::true_type {};


// tensor_is_primitive checks if a tensor includes a Data() member function thatJ returns an
// artihmetic pointer type.
template <typename TTensor> using tensor_data_return_type = decltype(std::declval<TTensor>().Data());

template<class TTensor> struct tensor_is_primitive_helper :
  std::integral_constant<bool, std::is_pointer_v<tensor_data_return_type<TTensor>> &&
                               std::is_arithmetic_v<std::remove_pointer_t<tensor_data_return_type<TTensor>>>> {};

template<class TTensor>
struct tensor_is_primitive : tensor_is_primitive_helper<std::remove_reference_t<TTensor>> {};

//
// Concepts
//

/// Arithmetic defines an arithmetic type, such as integer, float, etc.
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/// AnyTensor requires that the provided argument is a tensor.
template <typename TTensor>
concept AnyTensor = is_tensor_v<TTensor>;

/// PrimitiveTensor requires that the buffer is directly accessible and consists of primitive types,
template <typename TTensor>
concept PrimitiveTensor = AnyTensor<TTensor> && tensor_is_primitive<TTensor>::value;

/// AnyOperator requires that the provided argument is a tensor operator.
template <typename TOperator>
concept AnyOperator = is_operator_v<TOperator>;

/// TensorConvertible requires that the tensor is a tensor or functor that returns a tensor
template <typename TTensor>
concept TensorConvertible = is_tensor_v<TTensor> || is_operator_v<TTensor>;

// FIXME doesn't work for Views directly, assuming it converts View to Tensor before??
template <typename TFrom, template <typename, size_t, typename...> typename TTensor>
struct tensor_is_convertible_to : std::is_assignable<TTensor<typename TFrom::value_type, TFrom::rank>, TFrom> {};

template <typename TFrom, template <typename, size_t, typename...> typename TTensor>
concept ConvertibleTo = std::is_class_v<std::remove_cvref_t<TFrom>> && tensor_is_convertible_to<std::remove_cvref_t<TFrom>, TTensor>::value;

//
// Tensor basic arithmetic operations
//

// The arithmetic declaraions must be specialized for different tensor types, which supports
// specializations for accelerators.

template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorAdd;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorMatMul;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorElemMul;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorRmsNorm;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorSoftMax;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorSilu;

//
// Arithmentic operator overloading
//

// operator+ (TensorType, TensorType)
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto operator+(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return TensorAdd(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

// operator* (TensorType, TensorType) -> ElemMul, requires same rank
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto operator*(TTensor1&& tensor1, TTensor2&& tensor2)
requires (std::decay_t<TTensor1>::rank == std::decay_t<TTensor2>::rank)
{
  return TensorElemMul(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

// operator* (TensorType, TensorType) -> Scale, if one Tensor has rank0
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto operator*(TTensor1&& tensor1, TTensor2&& tensor2)
requires (std::decay_t<TTensor1>::rank == 0 || std::decay_t<TTensor2>::rank == 0)
{
  return TensorMatMul(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

// operator* (TensorType, arithmetic)
template <TensorConvertible TTensor, Arithmetic T>
auto operator*(TTensor&& tensor, T scalar)
{
  return TensorMatMul(std::forward<TTensor>(tensor), scalar);
}

// operator* (arithmetic, TensorType)
template <Arithmetic T, TensorConvertible TTensor>
auto operator*(T scalar, TTensor&& tensor)
{
  return TensorMatMul(scalar, std::forward<TTensor>(tensor));
}

} // end of namespace grid

/// operator<<(TENSOR) overloads the output operator for tensors.
std::ostream& operator<<(std::ostream& os, const grid::AnyTensor auto& tensor)
{
  using value_type = typename std::remove_reference_t<decltype(tensor)>::value_type;
  size_t rank = std::remove_reference_t<decltype(tensor)>::rank;

  auto dimensions = tensor.Dimensions();
  auto strides = tensor.Strides();

  std::function<void(int, const value_type*&)> print;
  print = [&os, &dimensions, &strides, &print, &rank](size_t index, const value_type* ptr) {
    os << "{ ";
    if (index < rank -1)
    {
      for (size_t i = dimensions[index]; i > 0; i--)
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
      for (size_t i = dimensions[index]; i > 0; i--)
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
