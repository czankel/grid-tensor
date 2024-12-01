//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <benchmark/benchmark.h>

#include <grid/tensor/tensor.h>
#include <grid/tensor/mmap.h>

/// Base
#include <grid/tensor/base/tensor.h>
#include <grid/tensor/base/binary.h>

/// Metal
#if METAL
#include <grid/tensor/metal/tensor.h>
#include <grid/tensor/metal/binary.h>
#endif

#include "benchmark.h"


BENCHMARK_GRID_TENSOR_DEFINE_F(GridTensorFixture, BinaryAdd, typename, T, typename, D)(benchmark::State& state)
{
  Tensor tensor1({10000, 10000}, T{1});
  Tensor tensor2({10000, 10000}, T{2});
  for (auto _ : state)
  {
    Tensor result = tensor1 + tensor2;
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_GRID_TENSOR_REGISTER(GridTensorFixture, BinaryAdd, int, grid::device::Base);
BENCHMARK_GRID_TENSOR_REGISTER(GridTensorFixture, BinaryAdd, float, grid::device::Base);

#if METAL
BENCHMARK_GRID_TENSOR_REGISTER(GridTensorFixture, BinaryAdd, int, grid::device::Metal);
BENCHMARK_GRID_TENSOR_REGISTER(GridTensorFixture, BinaryAdd, float, grid::device::Metal);
#endif
