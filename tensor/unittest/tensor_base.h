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
  template <typename T, size_t N, typename... A>
  using Tensor = grid::Tensor<T, N, A...>;
};

#else

struct TensorBaseType
{
  template <typename T, size_t _Rank, typename... _Allocator>
  class Tensor : public grid::Tensor<T, _Rank, _Allocator...>
  {
   public:
    using grid::Tensor<T, _Rank, _Allocator...>::Tensor;
  };

  // rank-0 tensor
  template <typename T>
  explicit Tensor(T) -> Tensor<T, 0>;
  template <typename T>
  explicit Tensor(grid::Uninitialized<T>) -> Tensor<T, 0>;

  // static tensors
  template <typename T, typename... Ts>
  explicit Tensor(T, Ts...) -> Tensor<std::common_type_t<T, Ts...>, 1, grid::StaticAllocator<sizeof...(Ts)+1>>;
  template <typename T, size_t... N>
  explicit Tensor(T(&&... l)[N]) -> Tensor<T, 2, grid::StaticAllocator<sizeof...(N), std::max({N...})>>;
  template <typename T, size_t... M, size_t... N>
  explicit Tensor(T(&&... l)[M][N]) -> Tensor<T, 3, grid::StaticAllocator<sizeof...(M), std::max({M...}), std::max({N...})>>;

  // dynamic tensors
  template <typename T>
  explicit Tensor(size_t, T) -> Tensor<T, 1>;
  template <typename T>
  explicit Tensor(size_t, grid::Uninitialized<T>) -> Tensor<T, 1>;
  template <typename T>
  explicit Tensor(size_t, size_t, T) -> Tensor<T, 2>;
  template <typename T>
  explicit Tensor(size_t, size_t, grid::Uninitialized<T>) -> Tensor<T, 2>;
/// >>>>>>
//
//
//
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], const ssize_t(&)[N], T) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], const ssize_t(&)[N], grid::Uninitialized<T>) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(size_t(&&)[N], ssize_t(&&)[N], grid::Uninitialized<T>) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], T) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], grid::Uninitialized<T>) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&&)[N], T) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&&)[N], grid::Uninitialized<T>) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, T) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, grid::Uninitialized<T>) -> Tensor<T, N>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, std::array<ssize_t, N>, grid::Uninitialized<T>) -> Tensor<T, N>;

  // array-view tensors
  template <typename _Tp, size_t N>
  explicit Tensor(grid::ArrayView<_Tp, N>) -> Tensor<_Tp, N, grid::NoAllocator>;

  // copy & move constructors
  template <typename T, size_t N, typename _Allocator>
  Tensor(const grid::Tensor<T, N, _Allocator>&) -> Tensor<T, N>;
  template <typename T, size_t N>
  Tensor(grid::Tensor<T, N>&&) -> Tensor<T, N>;

  // tensor view
  template <template <typename, size_t> typename TensorView, typename _Tensor, size_t _Rank>
  Tensor(TensorView<_Tensor, _Rank>&&) -> Tensor<typename _Tensor::value_type, _Rank>;
  template <template <typename, size_t> typename TensorView, typename _Tensor, size_t _Rank>
  Tensor(const TensorView<_Tensor, _Rank>&) -> Tensor<typename _Tensor::value_type, _Rank>;

  //  operators
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename Operator,
  template <typename, size_t, typename...> typename _Tensor, typename T, size_t _Rank, typename... Tensors>
  Tensor(Operator<_Tensor, T, _Rank, Tensors...>&&) -> Tensor<T, _Rank>;
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename Operator,
  template <typename, size_t, typename...> typename _Tensor, typename T, size_t _Rank, typename... Tensors>
  Tensor(const Operator<_Tensor,T,  _Rank, Tensors...>&) -> Tensor<T, _Rank>;
};

#endif
