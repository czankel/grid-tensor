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
  template <typename T, size_t N, typename... TAllocator>
  using Tensor = grid::Tensor<T, N, TAllocator...>;
};

#else

struct TensorBaseType
{
  template <typename T, size_t TRank, typename... TAllocator>
  class Tensor : public grid::Tensor<T, TRank, TAllocator...>
  {
   public:
    using grid::Tensor<T, TRank, TAllocator...>::Tensor;
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

  // memory-mapped tensors
  template <grid::Scalar T, size_t N>
  explicit Tensor(const size_t(&)[N], std::tuple<T*, size_t>&) -> Tensor<T, N, grid::NoAllocator>;
  template <grid::Scalar T, size_t N>
  explicit Tensor(const size_t(&)[N], std::tuple<T*, size_t>&&) -> Tensor<T, N, grid::NoAllocator>;

  // copy & move constructors
  template <typename T, size_t N, typename TAllocator>
  Tensor(const grid::Tensor<T, N, TAllocator>&) -> Tensor<T, N>;
  template <typename T, size_t N>
  Tensor(grid::Tensor<T, N>&&) -> Tensor<T, N>;

  // tensor view
  template <template <typename, size_t> typename TensorView, typename TTensor, size_t TRank>
  Tensor(TensorView<TTensor, TRank>&&) -> Tensor<typename TTensor::value_type, TRank>;
  template <template <typename, size_t> typename TensorView, typename TTensor, size_t TRank>
  Tensor(const TensorView<TTensor, TRank>&) -> Tensor<typename TTensor::value_type, TRank>;

  //  operators
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename Operator,
  template <typename, size_t, typename...> typename TTensor, typename T, size_t TRank, typename... TTensors>
  Tensor(Operator<TTensor, T, TRank, TTensors...>&&) -> Tensor<T, TRank>;
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename Operator,
  template <typename, size_t, typename...> typename TTensor, typename T, size_t TRank, typename... TTensors>
  Tensor(const Operator<TTensor, T,  TRank, TTensors...>&) -> Tensor<T, TRank>;
};

#endif
