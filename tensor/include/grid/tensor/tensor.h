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

/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename> struct Uninitialized {};

struct TensorBase;

// Helper function that requires a tensor of any type
template <typename _Tensor>
inline constexpr bool is_tensor_v = std::is_base_of_v<TensorBase, std::remove_cvref_t<_Tensor>>;


// helper functions to identify if a Tensor is for a specific runtime
template <typename, template <size_t, typename, auto...> typename>
struct is_same_runtime : std::false_type {};

template <template <size_t, typename, auto...> typename _Tensor, size_t _Rank, typename _T, auto... _Args>
struct is_same_runtime<_Tensor<_Rank, _T, _Args...>, _Tensor> : std::true_type {};

template <typename _Tensor, template <size_t, typename, auto...> typename _TensorRT>
inline constexpr bool is_same_runtime_v = is_same_runtime<std::remove_cvref_t<_Tensor>, _TensorRT>::value;


// Concept for requiring a TensorType

template <typename _Tensor> concept TensorType = is_tensor_v<_Tensor>;


/// TensorBase provides a base class for derived "runtime" tensor implementations with
/// optimizations specific to CPUs and accelerators.
struct TensorBase
{
  /// operator<< outputs the tensor buffer.
  inline friend std::ostream& operator<<(std::ostream& os, const TensorType auto& tensor)
  {
    using value_type = std::remove_reference_t<decltype(tensor)>::value_type;
    constexpr size_t rank = tensor.Rank();

    std::function<void(int, const value_type*&)> print;
    print = [&os, &tensor, &print](size_t index, const value_type*& ptr) {
      os << "{ ";
      if (index < rank - 1)
      {
        for (size_t i = tensor.Dim(index); i > 0; i--)
        {
          print(index + 1, ptr);
          if (i != 1)
            os << ", ";
          else
            os << " }";
        }
      }
      else
      {
        for (size_t i = tensor.Dim(index); i > 0; i--)
        {
          os << *ptr++;
          if (i != 1)
            os << ", ";
          else
            os << " }";
        }
      }
    };
    const value_type* ptr = tensor.Data();
    print(0, ptr);
    os << std::flush;

    return os;
  }
};

// Concepts to require a tensor to be of a specific type and rank.

template <typename _Tensor> concept TensorR0Type = is_tensor_v<_Tensor> && _Tensor::Rank() == 0;
template <typename _Tensor> concept TensorR1Type = is_tensor_v<_Tensor> && _Tensor::Rank() == 1;
template <typename _Tensor> concept TensorR2Type = is_tensor_v<_Tensor> && _Tensor::Rank() == 2;
template <typename _Tensor> concept TensorR3Type = is_tensor_v<_Tensor> && _Tensor::Rank() == 3;


} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_H
