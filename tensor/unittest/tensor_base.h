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

  // rank-0 tensor
  template <typename _T>
  explicit Tensor(_T) -> Tensor<_T, 0>;
  template <typename _T>
  explicit Tensor(grid::Uninitialized<_T>) -> Tensor<_T, 0>;

  // static tensors
  template <typename _T, typename... _Ts>
  explicit Tensor(_T, _Ts...) -> Tensor<std::common_type_t<_T, _Ts...>, 1, grid::StaticAllocator<sizeof...(_Ts)+1>>;
  template <typename _T, size_t... _N>
  explicit Tensor(_T(&&... l)[_N]) -> Tensor<_T, 2, grid::StaticAllocator<sizeof...(_N), std::max({_N...})>>;

  // dynamic tensors
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

  // tensor array-view
  template <typename _Tp, size_t N>
  explicit Tensor(grid::ArrayView<_Tp, N>) -> Tensor<_Tp, N, grid::NoAllocator>;

  // operators
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
  template <typename, size_t, typename...> typename _TensorRT, typename _T, size_t _Rank, typename... _Tensors>
  Tensor(_TensorOp<_TensorRT, _T, _Rank, _Tensors...>&&) -> Tensor<_T, _Rank>;
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename _TensorOp,
  template <typename, size_t, typename...> typename _TensorRT, typename _T, size_t _Rank, typename... _Tensors>
  Tensor(const _TensorOp<_TensorRT,_T,  _Rank, _Tensors...>&) -> Tensor<_T, _Rank>;
};

#endif
