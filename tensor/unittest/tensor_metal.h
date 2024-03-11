//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// Important: Class template argument deduction for alias templates P1814R0 not supported on all
// compilers. This requires to duplicate *all* deduction rules in slowcpu/tensor.h

#include <grid/tensor/concepts.h>

#if __cpp_deduction_guides >= 201907L

struct TensorMetalType
{
  template <typename T, size_t N>
  using DeviceTensor = grid::Tensor<T, N, device::Metal>

  // FIXME: this is basically grid::Tensor .. use with .. DeviceTensor 
  //FIXME: future could just be UseDevice(device::Metal); and Tensor a { ... };
  template <typename T, size_t N, typename M>
  using Tensor = grid::Tensor<T, N, M>;

  template <grid::AnyTensor T>
  DeviceTensor(T) -> DeviceTensor<typename std::remove_cvref_t<T>::value_type, std::remove_cvref_t<T>::rank>;
};

#else

struct TensorMetalType
{
#if 0
  template <typename T, size_t TRank>
  class DeviceTensor : public grid::Tensor<T, TRank, grid::DynamicMemory<grid::device::Metal>>
  {
   public:
    using grid::Tensor<T, TRank, grid::DynamicMemory<grid::device::Metal>>::Tensor;
  };
#endif
  template <typename T, size_t TRank, typename TMemory>
  class Tensor : public grid::Tensor<T, TRank, TMemory>
  {
   public:
    using grid::Tensor<T, TRank, TMemory>::Tensor;
  };

  template <grid::AnyTensor T>
  Tensor(T) -> Tensor<typename std::remove_cvref_t<T>::value_type,
                      std::remove_cvref_t<T>::rank,
                      grid::DynamicMemory<grid::device::Metal>>;

  // rank-0 tensor (cpu)
  template <typename T>
  explicit Tensor(T) -> Tensor<T, 0, grid::Scalar>;
  template <typename T>
  explicit Tensor(grid::Uninitialized<T>) -> Tensor<T, 0, grid::Scalar>;

  // static tensors (cpu)
  template <typename T, typename... Ts>
  explicit Tensor(T, Ts...) -> Tensor<std::common_type_t<T, Ts...>, 1, grid::StaticMemory<sizeof...(Ts)+1>>;
  template <typename T, size_t... N>
  explicit Tensor(T(&&... l)[N]) -> Tensor<T, 2, grid::StaticMemory<sizeof...(N), std::max({N...})>>;
  template <typename T, size_t... M, size_t... N>
  explicit Tensor(T(&&... l)[M][N]) -> Tensor<T, 3, grid::StaticMemory<sizeof...(M), std::max({M...}), std::max({N...})>>;


  // dynamic tensors (metal)
  template <typename T>
  explicit Tensor(size_t, T) -> Tensor<T, 1, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T>
  explicit Tensor(size_t, grid::Uninitialized<T>) -> Tensor<T, 1, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T>
  explicit Tensor(size_t, size_t, T) -> Tensor<T, 2, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T>
  explicit Tensor(size_t, size_t, grid::Uninitialized<T>) -> Tensor<T, 2, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], const ssize_t(&)[N], T) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], const ssize_t(&)[N], grid::Uninitialized<T>) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(size_t(&&)[N], ssize_t(&&)[N], grid::Uninitialized<T>) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], T) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], grid::Uninitialized<T>) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&&)[N], T) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&&)[N], grid::Uninitialized<T>) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, T) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, grid::Uninitialized<T>) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, std::array<ssize_t, N>, grid::Uninitialized<T>) -> Tensor<T, N, grid::DynamicMemory<grid::device::Metal>>;

  // Operator
  template <grid::AnyOperator TOperator>
  Tensor(const TOperator&) -> Tensor<typename TOperator::value_type, TOperator::rank, grid::DynamicMemory<grid::device::Metal>>;


};

#endif
