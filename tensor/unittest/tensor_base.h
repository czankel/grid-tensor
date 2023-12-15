//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// Important: Class template argument deduction for alias templates P1814R0 not supported on all
// compilers. This requires to duplicate *all* deduction rules in slowcpu/tensor.h


#if __cpp_deduction_guides >= 201907L

struct TensorBaseType
{
  template <typename _Tp, size_t _N, typename... _Allocator>
  using Tensor = grid::Tensor<_Tp, _N, _Allocator...>;
};

#else

struct TensorBaseType
{
  template <typename _Tp, size_t _Rank, typename... _Allocator>
  class Tensor : public grid::Tensor<_Tp, _Rank, _Allocator...>
  {
   public:
    using grid::Tensor<_Tp, _Rank, _Allocator...>::Tensor;
  };

  // rank-0 tensor
  template <typename _Tp>
  explicit Tensor(_Tp) -> Tensor<_Tp, 0>;
  template <typename _Tp>
  explicit Tensor(grid::Uninitialized<_Tp>) -> Tensor<_Tp, 0>;

  // static tensors
  template <typename _Tp, typename... _Ts>
  explicit Tensor(_Tp, _Ts...) -> Tensor<std::common_type_t<_Tp, _Ts...>, 1, grid::StaticAllocator<sizeof...(_Ts)+1>>;
  template <typename _Tp, size_t... _N>
  explicit Tensor(_Tp(&&... l)[_N]) -> Tensor<_Tp, 2, grid::StaticAllocator<sizeof...(_N), std::max({_N...})>>;
  template <typename _Tp, size_t... M, size_t... _N>
  explicit Tensor(_Tp(&&... l)[M][_N]) -> Tensor<_Tp, 3, grid::StaticAllocator<sizeof...(M), std::max({M...}), std::max({_N...})>>;

  // dynamic tensors
  template <typename _Tp>
  explicit Tensor(size_t, _Tp) -> Tensor<_Tp, 1>;
  template <typename _Tp>
  explicit Tensor(size_t, grid::Uninitialized<_Tp>) -> Tensor<_Tp, 1>;
  template <typename _Tp>
  explicit Tensor(size_t, size_t, _Tp) -> Tensor<_Tp, 2>;
  template <typename _Tp>
  explicit Tensor(size_t, size_t, grid::Uninitialized<_Tp>) -> Tensor<_Tp, 2>;
  template <typename _Tp, size_t _N>
  explicit Tensor(const size_t(&)[_N], const ssize_t(&)[_N], _Tp) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(const size_t(&)[_N], const ssize_t(&)[_N], grid::Uninitialized<_Tp>) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(size_t(&&)[_N], ssize_t(&&)[_N], _Tp) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(size_t(&&)[_N], ssize_t(&&)[_N], grid::Uninitialized<_Tp>) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(const size_t(&)[_N], _Tp) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(const size_t(&)[_N], grid::Uninitialized<_Tp>) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(const size_t(&&)[_N], _Tp) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(const size_t(&&)[_N], grid::Uninitialized<_Tp>) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(std::array<size_t, _N>, _Tp) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(std::array<size_t, _N>, std::array<ssize_t, _N>, _Tp) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(std::array<size_t, _N>, grid::Uninitialized<_Tp>) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  explicit Tensor(std::array<size_t, _N>, std::array<ssize_t, _N>, grid::Uninitialized<_Tp>) -> Tensor<_Tp, _N>;

  // array-view tensors
  template <typename _Tp, size_t _N>
  explicit Tensor(grid::ArrayView<_Tp, _N>) -> Tensor<_Tp, _N, grid::NoAllocator>;

  // copy & move constructors
  template <typename _Tp, size_t _N, typename _Allocator>
  Tensor(const grid::Tensor<_Tp, _N, _Allocator>&) -> Tensor<_Tp, _N>;
  template <typename _Tp, size_t _N>
  Tensor(grid::Tensor<_Tp, _N>&&) -> Tensor<_Tp, _N>;

  // tensor view
  template <template <typename, size_t> typename TensorView, typename _Tensor, size_t _Rank>
  Tensor(TensorView<_Tensor, _Rank>&&) -> Tensor<typename _Tensor::value_type, _Rank>;
  template <template <typename, size_t> typename TensorView, typename _Tensor, size_t _Rank>
  Tensor(const TensorView<_Tensor, _Rank>&) -> Tensor<typename _Tensor::value_type, _Rank>;

  //  operators
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename Operator,
  template <typename, size_t, typename...> typename _Tensor, typename _Tp, size_t _Rank, typename... _Tensors>
  Tensor(Operator<_Tensor, _Tp, _Rank, _Tensors...>&&) -> Tensor<_Tp, _Rank>;
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename Operator,
  template <typename, size_t, typename...> typename _Tensor, typename _Tp, size_t _Rank, typename... _Tensors>
  Tensor(const Operator<_Tensor, _Tp,  _Rank, _Tensors...>&) -> Tensor<_Tp, _Rank>;
};

#endif
