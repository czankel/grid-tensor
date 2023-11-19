//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// Class template argument deduction for alias templates P1814R0 not supported yet on all compilers
// ... requires to duplicate all deduction rules.


#if __cpp_deduction_guides >= 201907L

struct TensorBaseType
{
  template <typename _T, size_t _Rank, typename _Allocator>
  using Tensor = grid::Tensor<_T, _Rank, _Allocator>
};

#else

struct TensorBaseType
{
  template <typename _T, size_t _Rank, typename _Allocator = std::allocator<_T>>
  class Tensor : public grid::Tensor<_T, _Rank, _Allocator>
  {
    using grid::Tensor<_T, _Rank, _Allocator>::Tensor;
  };

  // Rank-0
  template <typename _T>
  explicit Tensor(_T) -> Tensor<_T, 0>;
  template <typename _T>
  explicit Tensor(grid::Uninitialized<_T>) -> Tensor<_T, 0>;

  // Tensor with Static Allocator - Brace-Initialization
  template <typename _T, typename... _Ts>
  explicit Tensor(_T, _Ts...) -> Tensor<std::common_type_t<_T, _Ts...>, 1, grid::StaticAllocator<sizeof...(_Ts)+1>>;
  template <typename _T, size_t... _N>
  explicit Tensor(_T(&&... l)[_N]) -> Tensor<_T, 2, grid::StaticAllocator<sizeof...(_N), std::max({_N...})>>;
  template <typename T, size_t... M, size_t... N>
  explicit Tensor(T(&&... l)[M][N]) -> Tensor<T, 3, grid::StaticAllocator<sizeof...(M), std::max({M...}), std::max({N...})>>;

  // Tensor with Dynamic Allocator - Parameters
  template <typename _T>
  explicit Tensor(size_t, _T) -> Tensor<_T, 1>;
  template <typename _T>
  explicit Tensor(size_t, grid::Uninitialized<_T>) -> Tensor<_T, 1>;
  template <typename _T>
  explicit Tensor(size_t, size_t, _T) -> Tensor<_T, 2>;
  template <typename _T>
  explicit Tensor(size_t, size_t, grid::Uninitialized<_T>) -> Tensor<_T, 2>;
  template <typename _T, size_t _N>
  explicit Tensor(const size_t(&)[_N], const ssize_t(&)[_N], _T) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(const size_t(&)[_N], const ssize_t(&)[_N], grid::Uninitialized<_T>) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(size_t(&&)[_N], ssize_t(&&)[_N], _T) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(size_t(&&)[_N], ssize_t(&&)[_N], grid::Uninitialized<_T>) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(const size_t(&)[_N], _T) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(const size_t(&)[_N], grid::Uninitialized<_T>) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(const size_t(&&)[_N], _T) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(const size_t(&&)[_N], grid::Uninitialized<_T>) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(std::array<size_t, _N>, _T) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(std::array<size_t, _N>, std::array<ssize_t, _N>, _T) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(std::array<size_t, _N>, grid::Uninitialized<_T>) -> Tensor<_T, _N>;
  template <typename _T, size_t _N>
  explicit Tensor(std::array<size_t, _N>, std::array<ssize_t, _N>, grid::Uninitialized<_T>) -> Tensor<_T, _N>;

  // Tensor with Dynamic Allocator - TensorView
  template <template <typename, size_t> typename TensorView, typename _Tensor, size_t _Rank>
  Tensor(TensorView<_Tensor, _Rank>&&) -> Tensor<typename _Tensor::value_type, _Rank>;
  template <template <typename, size_t> typename TensorView, typename _Tensor, size_t _Rank>
  Tensor(const TensorView<_Tensor, _Rank>&) -> Tensor<typename _Tensor::value_type, _Rank>;

  // Tensor with Dynamic Allocator - Copy and Move
  template <typename T, size_t N, typename _Allocator>
  Tensor(const grid::Tensor<T, N, _Allocator>&) -> Tensor<T, N>;
  template <typename T, size_t N>
  Tensor(grid::Tensor<T, N>&&) -> Tensor<T, N>;

  // Tensor with Dynamic Allocator - Operators
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
  template <typename, size_t, typename...> typename _TensorRT, typename _T, size_t _Rank, typename... _Tensors>
  Tensor(_TensorOp<_TensorRT, _T, _Rank, _Tensors...>&&) -> Tensor<_T, _Rank>;
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
  template <typename, size_t, typename...> typename _TensorRT, typename _T, size_t _Rank, typename... _Tensors>
  Tensor(const _TensorOp<_TensorRT,_T,  _Rank, _Tensors...>&) -> Tensor<_T, _Rank>;

  // Tensor with "NoAllocator" - ArrayView
  template <typename _T, size_t N>
  explicit Tensor(grid::ArrayView<_T, N>) -> Tensor<_T, N, grid::NoAllocator>;
};

#endif
