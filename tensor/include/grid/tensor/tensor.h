//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_H
#define GRID_TENSOR_TENSOR_H

//#include <functional>
//#include <iomanip>
#include <iostream>
//#include <numeric>

#include "concepts.h"
#include "tensor_helpers.h"

namespace grid {

constexpr static size_t kMaxRank = 4;

/// StaticAllocator is a special "allocator for constant static data.
template <size_t...> struct StaticAllocator {};

/// Broadcast defines to set the dimension to 1 ("broadcastable") in the axes argument of Tensor::View.
inline constexpr ssize_t Broadcast = -1;

/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename T> struct Uninitialized { using type = T; };


// Empty arithmetic operator declaraions that can be specialized for different tensor types.
// Specializing tensors enables also the use of the acellerators

template <template <typename, size_t, typename...> typename, typename, size_t, typename... > struct TensorAdd;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > struct TensorMul;
template <template <typename, size_t, typename...> typename, typename, size_t, typename... > struct TensorRmsNorm;


// Operator overloading

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
