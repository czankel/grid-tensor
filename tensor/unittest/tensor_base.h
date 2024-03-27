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
  template <typename T, size_t N, typename M>
  using Tensor = grid::Tensor<T, N, M>;
};

#else

struct TensorBaseType
{
  template <typename T, size_t TRank, typename TMemory>
  class Tensor : public grid::Tensor<T, TRank, TMemory>
  {
   public:
    using grid::Tensor<T, TRank, TMemory>::Tensor;
  };

  // rank-0 tensor
  template <typename T>
  explicit Tensor(T) -> Tensor<T, 0, grid::Scalar>;
  template <typename T>
  explicit Tensor(grid::Uninitialized<T>) -> Tensor<T, 0, grid::Scalar>;

  // static tensors
  template <typename T, typename... Ts>
  Tensor(T, Ts...) -> Tensor<std::common_type_t<T, Ts...>, 1, grid::StaticMemory<sizeof...(Ts)+1>>;
  template <typename T, size_t... N>
  Tensor(T(&&... l)[N]) -> Tensor<T, 2, grid::StaticMemory<sizeof...(N), std::max({N...})>>;
  template <typename T, size_t... M, size_t... N>
  Tensor(T(&&... l)[M][N]) -> Tensor<T, 3, grid::StaticMemory<sizeof...(M), std::max({M...}), std::max({N...})>>;
  // dynamic tensors
  template <typename T>
  explicit Tensor(size_t, T) -> Tensor<T, 1, grid::DeviceMemory<grid::device::Base>>;
  template <typename T>
  explicit Tensor(size_t, grid::Uninitialized<T>) -> Tensor<T, 1, grid::DeviceMemory<grid::device::Base>>;
  template <typename T>
  explicit Tensor(size_t, size_t, T) -> Tensor<T, 2, grid::DeviceMemory<grid::device::Base>>;
  template <typename T>
  explicit Tensor(size_t, size_t, grid::Uninitialized<T>) -> Tensor<T, 2, grid::DeviceMemory<grid::device::Base>>;
  template <typename T>
  explicit Tensor(size_t, size_t, size_t, T) -> Tensor<T, 3, grid::DeviceMemory<grid::device::Base>>;
  template <typename T>
  explicit Tensor(size_t, size_t, size_t, grid::Uninitialized<T>) -> Tensor<T, 3, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], const ssize_t(&)[N], T) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], const ssize_t(&)[N], grid::Uninitialized<T>) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(size_t(&&)[N], ssize_t(&&)[N], grid::Uninitialized<T>) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], T) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&)[N], grid::Uninitialized<T>) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&&)[N], T) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(const size_t(&&)[N], grid::Uninitialized<T>) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, T) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, grid::Uninitialized<T>) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N>
  explicit Tensor(std::array<size_t, N>, std::array<ssize_t, N>, grid::Uninitialized<T>) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;


  // memory-mapped tensors
  template <grid::Arithmetic T, size_t N>
  explicit Tensor(const size_t(&)[N], const std::tuple<T*, size_t>&) -> Tensor<T, N, grid::MemoryMapped>;
  template <grid::Arithmetic T, size_t N>
  explicit Tensor(const std::array<size_t, N>&, const std::tuple<T*, size_t>&) -> Tensor<T, N, grid::MemoryMapped>;

  // copy & move constructors
  template <typename T, size_t N, typename M>
  Tensor(const grid::Tensor<T, N, M>&) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;
  template <typename T, size_t N, typename M>
  Tensor(grid::Tensor<T, N, M>&&) -> Tensor<T, N, grid::DeviceMemory<grid::device::Base>>;

  // tensor view
  template <template <typename, size_t> typename TensorView, typename TTensor, size_t TRank>
  Tensor(TensorView<TTensor, TRank>&&) -> Tensor<typename TTensor::value_type, TRank, grid::DeviceMemory<grid::device::Base>>;
  template <template <typename, size_t> typename TensorView, typename TTensor, size_t TRank>
  Tensor(const TensorView<TTensor, TRank>&) -> Tensor<typename TTensor::value_type, TRank, grid::DeviceMemory<grid::device::Base>>;

  // operators
  template <grid::AnyOperator TOperator>
  Tensor(const TOperator&) ->
    Tensor<typename TOperator::value_type, TOperator::rank, grid::DeviceMemory<grid::device::Base>>;
};

#endif
