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
  template <typename T, size_t _Rank, typename _Allocator>
  using Tensor = grid::base::Tensor<T, _Rank, _Allocator>;
};

#else

namespace grid {

struct TensorBaseType
{
  template <typename T, size_t _Rank, typename _Allocator = std::allocator<T>>
  class Tensor : public grid::base::Tensor<T, _Rank, _Allocator>
  {
   public:
    using grid::base::Tensor<T, _Rank, _Allocator>::Tensor;
  };

  template <typename T>
  explicit Tensor(T) -> Tensor<T, 0>;
  template <typename T>
  explicit Tensor(grid::Uninitialized<T>) -> Tensor<T, 0>;
  template <typename T, typename... Ts>
  explicit Tensor(T, Ts...) -> Tensor<std::common_type_t<T, Ts...>, 1, grid::StaticAllocator<sizeof...(Ts)+1>>;
  template <typename T, size_t... _N>
  explicit Tensor(T(&&... l)[_N]) -> Tensor<T, 2, grid::StaticAllocator<sizeof...(_N), std::max({_N...})>>;
  template <typename T, size_t... _M, size_t... _N>
  explicit Tensor(T(&&... l)[_M][_N]) -> Tensor<T, 3, grid::StaticAllocator<sizeof...(_M), std::max({_M...}), std::max({_N...})>>;
  template <typename T>
  explicit Tensor(size_t, T) -> Tensor<T, 1>;
  template <typename T>
  explicit Tensor(size_t, grid::Uninitialized<T>) -> Tensor<T, 1>;
  template <typename T>
  explicit Tensor(size_t, size_t, T) -> Tensor<T, 2>;
  template <typename T>
  explicit Tensor(size_t, size_t, grid::Uninitialized<T>) -> Tensor<T, 2>;
  template <typename T, size_t _N>
  explicit Tensor(const size_t(&)[_N], const ssize_t(&)[_N], T) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(const size_t(&)[_N], const ssize_t(&)[_N], grid::Uninitialized<T>) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(size_t(&&)[_N], ssize_t(&&)[_N], T) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(size_t(&&)[_N], ssize_t(&&)[_N], grid::Uninitialized<T>) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(const size_t(&)[_N], T) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(const size_t(&)[_N], grid::Uninitialized<T>) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(const size_t(&&)[_N], T) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(const size_t(&&)[_N], grid::Uninitialized<T>) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(std::array<size_t, _N>, T) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(std::array<size_t, _N>, std::array<ssize_t, _N>, T) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(std::array<size_t, _N>, grid::Uninitialized<T>) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(std::array<size_t, _N>, std::array<ssize_t, _N>, grid::Uninitialized<T>) -> Tensor<T, _N>;
  template <typename T, size_t _N>
  explicit Tensor(grid::MMapArray<T, _N>) -> Tensor<T, _N, grid::MemoryMapped<T>>;
  // copy & move constructors FIXME: cannot get it wo rork
  template <typename T, size_t _N, typename _Allocator>
  Tensor(const grid::base::Tensor<T, _N, _Allocator>&) -> Tensor<T, _N>;
  //template <typename T, size_t _N>
  //explicit Tensor(grid::Tensor<T, _N>&&) -> Tensor<T, _N>;

  // FIXME: what are these?
  template <typename T, size_t _N>
  explicit Tensor(Tensor<T, 1, grid::StaticAllocator<_N>>&) -> Tensor<T, _N>;
  template <typename T, size_t _N, size_t _M>
  explicit Tensor(Tensor<T, 2, grid::StaticAllocator<_N, _M>>&) -> Tensor<T, _N * _M>;
  template <typename T, size_t _C, size_t _N, size_t _M>
  explicit Tensor(Tensor<T, 3, grid::StaticAllocator<_C, _N, _M>>&) -> Tensor<T, _C * _N * _M>;

  // tensorview FIXME: would this be better as a concept? template <ViewType...>
  template <template <typename, size_t> typename TensorView, typename _Tensor, size_t _Rank> // FIXME: use concept
  Tensor(TensorView<_Tensor, _Rank>&&) -> Tensor<typename _Tensor::value_type, _Rank>;
  template <template <typename, size_t> typename TensorView, typename _Tensor, size_t _Rank> // FIXME: use concept
  Tensor(const TensorView<_Tensor, _Rank>&) -> Tensor<typename _Tensor::value_type, _Rank>;
  // tensor-op
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename TensorOp,
  template <typename, size_t, typename...> typename TensorRT, typename T, size_t _Rank, typename... Tensors>
  Tensor(TensorOp<TensorRT, T, _Rank, Tensors...>&&) -> Tensor<T, _Rank>;
  template <template <template <typename, size_t, typename...> typename, typename, size_t, typename...> typename TensorOp,
  template <typename, size_t, typename...> typename TensorRT, typename T, size_t _Rank, typename... Tensors>
  Tensor(const TensorOp<TensorRT,T,  _Rank, Tensors...>&) -> Tensor<T, _Rank>;
};

} // end of namespace grid

#endif
