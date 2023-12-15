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

/// Tensor is an implementation of an "AI Tensor" for vector and matrix operations, and follows
/// less the mathematical or physical definition.
///
/// @tparam _Tp         Integral type
/// @tparam _Rank       Rank of the tensor with 0: scalar, 1: vector, 2: matrix, etc.
/// @tparam _Allocator  Stateless allocator; defaults to std::allocator
///
///
/// Tensors define these member types:
///
///   value_type        _Tp
///   allocator_type    _Allocator
///   pointer           Pointer type; depends on the implementation
///   const_pointer     Constant pointer type; depends on the implementation
///
///
/// Tensors also provide the following member methods:
///
///   constexpr size_t           Rank()
///   std::array<size_t, Rank>   Dimensions() const
///   std::array<ssize_t, Rank>  Strides() const
///   pointer                    Data()
///   const_pointer              Data() const
///
///
/// Tensor View
///
///   Tensor views are tensors that don't have "ownership" of the data but provide a "view" into
///   the underlying tensor. Because of the object lifetime of the tensor, it comes with restrictions:
///
///    - A view is not default constructible and can only be assigned to a tensor (or other tensor view).
///    - If a tensor (or tensor view is assigned to a view, the data is always copied to the view,
///      even if the source object is a rvalue reference.
///
///
/// Allocators
///
///  Tensors include a template parameter for the "allocator". The concept expands the allocator
///  definition of the std library and includes allocators for statically allocated buffers
///  (constants), device memory, and a "view" into the buffer of another tensor.
///
///  The tensor includes the allocate_type typedef, which can be probe with std::uses_allocator
///
///  The "view" has specific objecct ownership and lifetime dependencies that comes with
///  the following restrictions:
///
///   - assigning a view to a tensor (or other view) *always* copies the data, even if the view
///     is an rvalue reference.
///   - a view cannot be declared..
///   - an rvalue-reference tensor input returns a tensor instead of a view
///   - as an rvalue-reference, the view must be copied to the destination tensor
///

namespace grid {

/// StaticAllocator is a special "allocator" for constant static data.
template <size_t...> struct StaticAllocator {};

/// NoAllocator is a spcial "allocator" for an externally managed buffer.
struct NoAllocator {};

/// Broadcast defines to set the dimension to 1 ("broadcastable") in the axes argument of Tensor::View.
inline constexpr ssize_t Broadcast = -1;

/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename> struct Uninitialized {};

/// Scalar defines a number.
template <typename T>
concept Scalar = std::is_arithmetic_v<T>;

template<size_t _N>
struct Axes
{
  template<typename... _Tps>
  Axes(_Tps&& ... values) : data_{std::move(values)...} {}
  size_t data_[_N];
};

//
// Tensor Traits
//

/// is_tensor_v checks if a type is a tensor, which requires to have the member functions:
/// Dimensions, Strides, and Data.
template <typename _Tensor>
inline constexpr bool is_tensor_v =
  std::is_class_v<typename std::remove_cvref_t<_Tensor>> &&
  requires (const _Tensor& t) { t.Rank(); t.Dimensions(); t.Strides(); t.Data(); };


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

#if 1
template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _Operator,
          template <typename, size_t, typename...> typename _Tensor,
          typename _Tp, size_t _Rank, typename... _Tensors>
struct to_tensor<_Operator<_Tensor, _Tp, _Rank, _Tensors...>>
{
  //using type = typename std::invoke_result_t<decltype(std::declval<_Operator<_Tensor, _Tp, _Rank, _Tensors...>(_Tensors&...)>.operator()())>;
  using type = decltype(std::declval<_Operator<_Tensor, _Tp, _Rank, _Tensors...>>().operator()());

  //using type = typename std::invoke_result_t<decltype(std::declval<_Operator<_Tensor, _Tp, _Rank, _Tensors...>(_Tensors&...)>.operator()())>;//, _Operator>; //<_Tensor, _Tp, _Rank _Tensors...>>;
};

#else
template <typename _Operator> requires is_operator_v<_Operator>
struct to_tensor<_Operator>
{
  using type = typename std::invoke_result_t<decltype(std::declval<_Operator>().operator()), _Operator>;
};
#endif


// is_same_tensor<TENSOR1, TENSOR2> checks if two tensors are of the same type.
template <typename, template <typename, size_t, typename...> typename> struct is_same_tensor_as : std::false_type {};

template <template <typename, size_t, typename...> typename _Tensor, typename _Tp, size_t _Rank, typename... _Allocator>
struct is_same_tensor_as<_Tensor<_Tp, _Rank, _Allocator...>, _Tensor> : std::true_type {};


// tensor_is_convertible_to<FROM,TO> check if TO = FROM is a valid assignment.
template <typename, template <typename, size_t, typename...> typename> struct tensor_is_convertible_to;

template <template <typename, size_t, typename...> typename _Tensor1,
          template <typename, size_t, typename...> typename _Tensor2,
          typename _Tp, size_t _Rank, typename... _Allocator>
struct tensor_is_convertible_to<_Tensor1<_Tp, _Rank, _Allocator...>, _Tensor2>
 : std::is_assignable<_Tensor2<_Tp, _Rank>, _Tensor1<_Tp, _Rank, _Allocator...>>
{};

template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _Operator,
          template <typename, size_t, typename...> typename _Tensor,
          typename _Tp, size_t _Rank, typename... _Tensors>
struct tensor_is_convertible_to<_Operator<_Tensor, _Tp, _Rank, _Tensors...>, _Tensor>
 : std::is_assignable<_Tensor<_Tp, _Rank>, _Operator<_Tensor, _Tp, _Rank, _Tensors...>>
{};

#if 0// FIXME: doesn't work
template <template <typename, size_t, typename...> typename _Tensor,
          Scalar _Scalar,
          typename _Tp, size_t _Rank, typename... _Allocator>
struct tensor_is_convertible_to<_Tensor<_Tp, _Rank, _Allocator...>, _Scalar>
 : std::is_convertible<_Scalar, _Tensor<_Tp, _Rank>>
{};
#endif

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

template <typename T1, template <typename, size_t, typename...> typename T2>
concept ConvertibleTo = tensor_is_convertible_to<std::remove_cvref_t<T1>, T2>::value;

//
// Arithmetic function declarations
//

// The arithmetic declaraions must be specialized for different tensor types, which supports
// specializations for accelerators.

template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorAdd;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorMul;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorElemMul;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorRmsNorm;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorSoftMax;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > class TensorSilu;

//
// Arithmentic operator overloading
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

// operator* (TensorType, arithmetic)
template <TensorConvertible _Tensor, Scalar _Scalar>
auto operator*(_Tensor&& tensor, _Scalar scalar) // FIXME requires _Tensor::value_type == _Scalar
{
  return TensorMul(std::forward<_Tensor>(tensor), scalar);
}

// operator* (TensorType, arithmetic)
template <Scalar _Scalar, TensorConvertible _Tensor>
auto operator*(_Scalar scalar, _Tensor&& tensor)
{
  return TensorMul(scalar, std::forward<_Tensor>(tensor));
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
