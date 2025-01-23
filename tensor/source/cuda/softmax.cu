//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/precision.h>

#include <grid/tensor/cuda/device.h>
#include <grid/tensor/cuda/softmax.h>

#include "../instantiate.h"
#include "reduce.h"
#include "utils.h"

namespace grid {
namespace cuda {

template <typename T, unsigned int BlockSize>
__global__ void CudaSoftMax(T* d, const T* x, T eps, int dim)
{
  __shared__ T sdata[cuda::MaxThreadCount];

  unsigned int grid_size = gridDim.x * BlockSize;
  unsigned int tid = threadIdx.x;

  T max{0};  // FIXME: use T or float?
  // FIXME unroll (partially)
  for (unsigned int i = tid; i < dim; i += grid_size)
    max = x[i] > max ? x[i] : max;
  sdata[tid] = max;

  __syncthreads();

  CudaReduce<T, MaxOperator, BlockSize>(sdata, tid, dim);
  max = sdata[0];

  // d = exp(x - local-max); sum += d
  T sum{0};
  for (unsigned int i = tid; i < dim; i += grid_size)
  {
    d[i] = exp(x[i] - max);
    sum += d[i];
  }
  sdata[tid] = sum;

  __syncthreads();

  CudaReduce<T, AddOperator, BlockSize>(sdata, tid, dim);
  sum = sdata[0];
  T scale = static_cast<T>(1)/(sum + eps);
  printf("max %f sum %f\n", max, sum);

  __syncthreads();

  for (unsigned int i = tid; i < dim; i += grid_size)
    d[i] = d[i] * scale;
}

#if 0
template <typename T, unsigned int BlockSize>
__global__ void CudaSoftMax(T* d, const T* x, const T eps, int dim)
{
  __shared__ T sdata[cuda::MaxThreadCount];

  unsigned int tid_col = threadIdx.x;
  unsigned int tid_row = threadIdx.y * dim;
  unsigned int tid = tid_col + tid_row;
  unsigned int end = tid_col + dim;

  unsigned size_t block_start = blockIdx.y * blockDim.x + blockIdx.x * BlockSize;
  unsigned int grid_size = gridDim.x * BlockSize;
  x = x + block_start;

  T max{0};  // FIXME: use T or float?
  // FIXME unroll (partially)
  for (unsigned int i = tid; i < end; i += grid_size)
    max = std::max(max, x[i]);
  sdata[tid] = max;

  __syncthreads();

  max = CudaMapReduce(sdata, tid, end);

  // FIXME __syncthreads();

  // SumExp
  // exp(x - local-max)
  for (unsigned int i = idx_s; i < idx_e; i += grid_size)
  {
    d[i] = exp(x[i] - max);
    sdata[tid] += d[i];
  }

  //T sum = CudaAllReduce(sdata, tid_x, dim);
  //T scalar = static_cast<T>(1)/(sum + eps);

  //for (unsigned int i = idx_s; i < idx_e; i += grid_size)
    //d[i] = x[i] / scalar;
}
#endif

} // end of namespace cuda

template <typename T>
void SoftMaxCallKernel(T* d, const T* x, size_t rows, size_t cols, dim3 grid_size, dim3 block_size)
{
  size_t smem_size = cuda::MaxThreadCount;
  size_t n_threads = std::min(cols, 1024UL);  // FIXME: divide by sizeof(T) *2??
  int n_threads_log2 = sizeof(n_threads) * 8 - __builtin_clzll(n_threads - 1);

  T eps = Eps<T>::default_value;
printf("n-thread %d\n", n_threads_log2);
  switch (n_threads_log2)
  {
    #define CUDA_SOFTMAX_CASE(BIT) \
      case BIT: cuda::CudaSoftMax<T,1<<BIT><<<grid_size,block_size,smem_size>>>(d,x,eps,cols); break;
    INSTANTIATE1(CUDA_SOFTMAX_CASE, (9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
    default: throw std::runtime_error("invalid thread count in softmax");
  }
}


// note that lower ranks are contiguous
template <typename T, size_t R>
void SoftMaxOperator<device::Cuda>::EvalContiguous(
    T* d, const T* x,
    std::span<const size_t, R> dimensions,
    std::span<const ssize_t, R> strides_d,
    std::span<const ssize_t, R> strides_a) const
{
  if constexpr (R == 0)
  {
    printf("SF1\n");
    SoftMaxCallKernel(d, x, 1, 1, 1, 1);
  }
  else if constexpr (R == 1)
  {
    printf("SF2\n");
    auto [grid_size, block_size] = cuda::GetSizes(dimensions[0]);
    SoftMaxCallKernel(d, x, 1, dimensions[0], std::move(grid_size), std::move(block_size));
  }
  else if constexpr (R == 2)
  {
    printf("OPS1\n");
#if 0
    auto [block_size, grid_size] = cuda::GetSizes(dimensions, 16, 16);  // FIXME 256 threads instead of 1k? 32, 32?
    CudaSoftMaxContiguousRank2<T><<<block_size, grid_size>>>(
        d, a, cuda::MakeDim3(dimensions), cuda::MakeDim3(strides_d), cuda::MakeDim3(strides_a));
#endif
  }
  else if constexpr (R == 3)
  {
    printf("OPS1\n");
#if 0
    auto [block_size, grid_size] = cuda::GetSizes(dimensions, 8, 8, 8);   // FIXME 512 threads?
    CudaSoftMaxContiguousRank3<T><<<block_size, grid_size>>>(
        d, a, cuda::MakeDim3(dimensions), cuda::MakeDim3(strides_d), cuda::MakeDim3(strides_a));
#endif
  }
}


template <typename T, size_t R>
void SoftMaxOperator<device::Cuda>::EvalDiscontiguous(
    T* d, const T* a,
    std::span<const size_t, R> dimensions,
    std::span<const ssize_t, R> strides_d,
    std::span<const ssize_t, R> strides_a) const
{
  printf("DISCONTIGUOUS@!!\n");
}

#define FUNCTION_CONTIGUOUS(R, T) \
  template void SoftMaxOperator<device::Cuda>::EvalContiguous<T, R>( \
      T*, const T*, std::span<const size_t, R>, \
      std::span<const ssize_t, R>, std::span<const ssize_t, R>) const;

#define FUNCTION_DISCONTIGUOUS(R, T) \
  template void SoftMaxOperator<device::Cuda>::EvalDiscontiguous<T, R>( \
      T*, const T*,  std::span<const size_t, R>, \
      std::span<const ssize_t, R>, std::span<const ssize_t, R>) const;

#define TYPES  float
#define RANKS_CONTIGUOUS 0, 1, 2, 3
#define RANKS_DISCONTIGUOUS 1, 2, 3

INSTANTIATE2(FUNCTION_CONTIGUOUS, (RANKS_CONTIGUOUS), (TYPES))
INSTANTIATE2(FUNCTION_DISCONTIGUOUS, (RANKS_DISCONTIGUOUS), (TYPES))

} // end of namespace grid
