//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef TENSOR_BENCHMARK_BENCHMARK_H
#define TENSOR_BENCHMARK_BENCHMARK_H

#include "../source/instantiate.h"

#define TEMPLATE_ARG_0
#define TEMPLATE_ARG_2(TYPE, NAME) NAME
#define TEMPLATE_ARG_4(TYPE, NAME, ...) NAME, TEMPLATE_ARG_2(__VA_ARGS__)
#define TEMPLATE_ARG_6(TYPE, NAME, ...) NAME, TEMPLATE_ARG_4(__VA_ARGS__)
#define TEMPLATE_ARG_8(TYPE, NAME, ...) NAME, TEMPLATE_ARG_6(__VA_ARGS__)
#define TEMPLATE_ARG_10(TYPE, NAME, ...) NAME, TEMPLATE_ARG_8(__VA_ARGS__)
#define TEMPLATE_ARG_12(TYPE, NAME, ...) NAME, TEMPLATE_ARG_10(__VA_ARGS__)
#define TEMPLATE_ARG_14(TYPE, NAME, ...) NAME, TEMPLATE_ARG_12(__VA_ARGS__)
#define TEMPLATE_ARG_16(TYPE, NAME, ...) NAME, TEMPLATE_ARG_14(__VA_ARGS__)
#define TEMPLATE_ARG_18(TYPE, NAME, ...) NAME, TEMPLATE_ARG_16(__VA_ARGS__)
#define TEMPLATE_ARGS__(NARGS, ...) TEMPLATE_ARG_##NARGS(__VA_ARGS__)
#define TEMPLATE_ARGS_(NARGS, ...)  TEMPLATE_ARGS__(NARGS, __VA_ARGS__)
#define TEMPLATE_ARGS(...) TEMPLATE_ARGS_(PP_NARG(__VA_ARGS__), __VA_ARGS__)

#define TEMPLATE_PARAM_0
#define TEMPLATE_PARAM_2(TYPE, NAME) TYPE NAME
#define TEMPLATE_PARAM_4(TYPE, NAME, ...) TYPE NAME, TEMPLATE_PARAM_2(__VA_ARGS__)
#define TEMPLATE_PARAM_6(TYPE, NAME, ...) TYPE NAME, TEMPLATE_PARAM_4(__VA_ARGS__)
#define TEMPLATE_PARAM_8(TYPE, NAME, ...) TYPE NAME, TEMPLATE_PARAM_6(__VA_ARGS__)
#define TEMPLATE_PARAM_10(TYPE, NAME, ...) TYPE NAME, TEMPLATE_PARAM_8(__VA_ARGS__)
#define TEMPLATE_PARAM_12(TYPE, NAME, ...) TYPE NAME, TEMPLATE_PARAM_10(__VA_ARGS__)
#define TEMPLATE_PARAM_14(TYPE, NAME, ...) TYPE NAME, TEMPLATE_PARAM_12(__VA_ARGS__)
#define TEMPLATE_PARAM_16(TYPE, NAME, ...) TYPE NAME, TEMPLATE_PARAM_14(__VA_ARGS__)
#define TEMPLATE_PARAM_18(TYPE, NAME, ...) TYPE NAME, TEMPLATE_PARAM_16(__VA_ARGS__)
#define TEMPLATE_PARAMS__(NARGS, ...) TEMPLATE_PARAM_##NARGS(__VA_ARGS__)
#define TEMPLATE_PARAMS_(NARGS, ...)  TEMPLATE_PARAMS__(NARGS, __VA_ARGS__)
#define TEMPLATE_PARAMS(...) TEMPLATE_PARAMS_(PP_NARG(__VA_ARGS__), __VA_ARGS__)


//
// Extensions to the Google Benchmark
//


// BENCHMARK_GRID_TENSOR_FUNCTION(BaseClass, Method, ...)
// {
//   <test implementation>
// }

#define BENCHMARK_GRID_TENSOR_FUNCTION(BaseClass, Method, ...)                                    \
  template <TEMPLATE_PARAMS(__VA_ARGS__)>                                                         \
  class BENCHMARK_PRIVATE_CONCAT_NAME(BaseClass, Method) : public benchmark::Fixture              \
  {                                                                                               \
    template <size_t R>                                                                           \
    struct Tensor : public grid::Tensor<T, R, grid::DeviceMemory<D>>                              \
      { using grid::Tensor<T, R, grid::DeviceMemory<D>>::Tensor;                                  \
        using tensor_device = D; };                                                               \
    template <size_t N> Tensor(const size_t(&&)[N], T) -> Tensor<N>;                              \
    template <size_t N> Tensor(const size_t(&&)[N], grid::Uninitialized<T>) -> Tensor<N>;         \
    template <size_t N> Tensor(const size_t(&)[N], T) -> Tensor<N>;                               \
    template <size_t N> Tensor(const size_t(&)[N], grid::Uninitialized<T>) -> Tensor<N>;          \
    template <size_t N> Tensor(const size_t(&&)[N], const ssize_t(&&)[N], T) -> Tensor<N>;        \
    template <size_t N> Tensor(const size_t(&&)[N], const ssize_t(&&)[N], grid::Uninitialized<T>) \
      -> Tensor<N>;                                                                               \
    template <size_t N> Tensor(const size_t(&)[N], const ssize_t(&&)[N], T) -> Tensor<N>;         \
    template <size_t N> Tensor(const size_t(&)[N], const ssize_t(&&)[N], grid::Uninitialized<T>)  \
      -> Tensor<N>;                                                                               \
    template <size_t N> Tensor(std::array<size_t, N>, T) -> Tensor<N>;                            \
    template <size_t N> Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<N>;    \
    template <size_t N> Tensor(std::array<size_t, N>, grid::Uninitialized<T>) -> Tensor<N>;       \
    template <size_t N> Tensor(std::array<size_t, N>, std::array<ssize_t, N>, grid::Uninitialized<T>) \
      -> Tensor<N>;                                                                               \
    template <grid::AnyOperator TOperator> Tensor(const TOperator&) -> Tensor<TOperator::rank>;   \
    template <grid::AnyOperator TOperator> Tensor(TOperator&&) -> Tensor<TOperator::rank>;        \
    template <size_t N, typename M> Tensor(const grid::Tensor<T, N, M>&) -> Tensor<N>;            \
   public:                                                                                        \
    BENCHMARK_PRIVATE_CONCAT_NAME(BaseClass, Method)();                                           \
   protected:                                                                                     \
    void BenchmarkCase(::benchmark::State&) BENCHMARK_OVERRIDE;                                   \
  };

// Helper
#define BENCHMARK_GRID_TENSOR_DEFINE_F(BaseClass, Method, ...)                                    \
  BENCHMARK_GRID_TENSOR_FUNCTION(BaseClass, Method, __VA_ARGS__)                                  \
  template <TEMPLATE_PARAMS(__VA_ARGS__)>                                                         \
  void BENCHMARK_PRIVATE_CONCAT_NAME(BaseClass, Method)<TEMPLATE_ARGS(__VA_ARGS__)>::BenchmarkCase

// Helper
#define BENCHMARK_GRID_TENSOR_CONSTRUCTOR(BaseClass, Method, ...)                                 \
  template <>                                                                                     \
  BENCHMARK_PRIVATE_CONCAT_NAME(BaseClass, Method)<__VA_ARGS__>                                   \
  ::BENCHMARK_PRIVATE_CONCAT_NAME(BaseClass, Method)()                                            \
  {                                                                                               \
    this->SetName(stringify(BENCHMARK_GRID_TENSOR_NAME_F(BaseClass, Method, __VA_ARGS__)));       \
  }

// Helper
#define BENCHMARK_GRID_TENSOR_REGISTER_F(BaseClass, Method,  ...)                                 \
  BENCHMARK_PRIVATE_DECLARE(BENCHMARK_PRIVATE_CONCAT_NAME(BaseClass, Method)) =                   \
    (::benchmark::internal::RegisterBenchmarkInternal(                                            \
      new BENCHMARK_PRIVATE_CONCAT_NAME(BaseClass, Method)<__VA_ARGS__>()));

/// Register a templated test ... FIXME
#define BENCHMARK_GRID_TENSOR_REGISTER(BaseClass, Method, ...)                                    \
  BENCHMARK_GRID_TENSOR_CONSTRUCTOR(BaseClass, Method, __VA_ARGS__)                    \
  BENCHMARK_GRID_TENSOR_REGISTER_F(BaseClass, Method, __VA_ARGS__)

namespace grid {
  // extending tensor/concept.h to return the device from the tensor defined above
  template <typename TTensor>
  requires requires (TTensor::tensor_device t) {t;}
  struct tensor_device<TTensor> { using type = TTensor::tensor_device; };
}

#endif  // TENSOR_BENCHMARK_BENCHMARK_H
