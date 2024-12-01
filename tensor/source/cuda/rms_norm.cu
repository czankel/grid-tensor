//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <utility>

#include <grid/tensor/tensor.h>

#include <grid/tensor/cuda/binary.h>
#include <grid/tensor/cuda/device.h>

#include <grid/tensor/cuda/rms_norm.h>

#include "../instantiate.h"
#include "utils.h"


namespace grid {
namespace cuda {

#pragma diag_suppress 3012  // FIXME: doesn't work

template <typename T, unsigned int blockSize>
__device__ void CudaWarpReduce(volatile T *sdata, unsigned int tid, int dim)
{
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

// FIXME: use syncs...
// FIXME: is there any other blockSize???
template <typename T, unsigned int blockSize>
__global__ void CudaRmsNormLine(T* d, const T* x, const T eps, size_t dim)
{
  extern __shared__ T sdata[];
  __shared__ T rms; // FIXME: could use sdata[0]?

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockSize + tid;
  unsigned int grid_size = gridDim.x * blockSize;
  T sum = 0;  // FIXME: use T or float?

  for (unsigned int i = idx; i < dim; i += grid_size)
    sum += x[i] * x[i];

  sdata[tid] = sum;

  __syncthreads();

  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid + 64];  } __syncthreads(); }
  if (tid < 32) CudaWarpReduce<T, blockSize>(sdata, tid, dim);

  // FIXME: is x/sqrt(...) better or x * rsqrt(...)?
  if (tid == 0)
    rms = sqrt(sdata[0] / dim + eps);

  __syncthreads();

  for (unsigned int i = idx; i < dim; i += grid_size)
    d[i] = x[i] / rms;
}


template <typename T, unsigned int blockSize>
__global__ void CudaRmsNormLoop(T* d, const T* x, const T eps, dim3 dims)
{
  extern __shared__ T sdata[];
  __shared__ T rms; // FIXME: could use sdata[0]?

  if (threadIdx.y >= dims.y)
    return;

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockSize.x + threadIdx.y * dim  + threadIdx.x;
  unsigned int grid_size = gridDim.x * blockSize;
  T sum = 0;  // FIXME: use T or float?

  for (unsigned int i = idx; i < dim; i += grid_size)
    sum += x[i] * x[i];

  sdata[tid] = sum;

  __syncthreads();

  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid + 64];  } __syncthreads(); }
  if (tid < 32) CudaWarpReduce<T, blockSize>(sdata, tid, dim);

  // FIXME: is x/sqrt(...) better or x * rsqrt(...)?
  if (tid == 0)
    rms = sqrt(sdata[0] / dim + eps);

  __syncthreads();

  for (unsigned int i = idx; i < dim; i += grid_size)
    d[i] = x[i] / rms;
}

} // end of namespace cuda


template <typename T>
void RmsNormOperator<device::Cuda>::Eval(T* d, const T* a, const T eps, std::array<const size_t, 2>&& dims) const
{
  printf("Eval n-dim without weights\n");
  // TODO: what are the correct values here?
  auto [grid_size, block_size] = cuda::GetSizes(std::span(dims), 16, 16, 1);

  size_t smemSize = 1024; // FIXE this is fixed to the thread count * max type size??
  size_t n_threads = dim > 512? 512 : dim;  // FIXME: why 512? might come from original implementation that accessed consecutive blocks in one reduce op
  int n_threads_log2 = sizeof(n_threads) * 8 - __builtin_clzll(n_threads - 1);

  // FIXME: use c-macro?
  switch (n_threads_log2)
  {
    case 9 /* 512 */: cuda::CudaRmsNormLoop<T, 512><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 8 /* 256 */: cuda::CudaRmsNormLoop<T, 256><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 7 /* 128 */: cuda::CudaRmsNormLoop<T, 128><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 6 /*  64 */: cuda::CudaRmsNormLoop<T,  64><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 5 /*  32 */: cuda::CudaRmsNormLoop<T,  32><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 4 /*  16 */: cuda::CudaRmsNormLoop<T,  16><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 3 /*   8 */: cuda::CudaRmsNormLoop<T,   8><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 2 /*   4 */: cuda::CudaRmsNormLoop<T,   4><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 1 /*   2 */: cuda::CudaRmsNormLoop<T,   2><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 0 /*   1 */: cuda::CudaRmsNormLoop<T,   1><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    default: throw std::runtime_error("invalid thread count");
  }
}}

template <typename T>
void RmsNormOperator<device::Cuda>::Eval(T* d, const T* a, const T* w, const T eps, std::array<const size_t, 2>&& dims) const
{
  printf("Eval n-dim with weights\n");
}

template <typename T>
void RmsNormOperator<device::Cuda>::Eval(T* d, const T* x, const T eps, size_t dim) const
{
  auto [grid_size, block_size] = cuda::GetSizes(dim);

  size_t smemSize = 1024; // FIXE this is fixed to the thread count * max type size??
  size_t n_threads = dim > 512? 512 : dim;
  int n_threads_log2 = sizeof(n_threads) * 8 - __builtin_clzll(n_threads - 1);

  // FIXME: use c-macro?
  switch (n_threads_log2)
  {
    case 9 /* 512 */: cuda::CudaRmsNormLine<T, 512><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 8 /* 256 */: cuda::CudaRmsNormLine<T, 256><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 7 /* 128 */: cuda::CudaRmsNormLine<T, 128><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 6 /*  64 */: cuda::CudaRmsNormLine<T,  64><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 5 /*  32 */: cuda::CudaRmsNormLine<T,  32><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 4 /*  16 */: cuda::CudaRmsNormLine<T,  16><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 3 /*   8 */: cuda::CudaRmsNormLine<T,   8><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 2 /*   4 */: cuda::CudaRmsNormLine<T,   4><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 1 /*   2 */: cuda::CudaRmsNormLine<T,   2><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    case 0 /*   1 */: cuda::CudaRmsNormLine<T,   1><<< grid_size, block_size, smemSize >>>(d, x, eps, dim); break;
    default: throw std::runtime_error("invalid thread count");
  }
}

// FIXME: TOSO
template <typename T>
void RmsNormOperator<device::Cuda>::Eval(T* d, const T* x, const T* w, const T eps, size_t dim) const
{
#if 0
  auto [grid_size, block_size] = cuda::GetSizes(dim);

  size_t smemSize = 1024;
  size_t n_threads = dim > 512? 512 : dim;
  int n_threads_log2 = sizeof(n_threads) * 8 - __builtin_clzll(n_threads - 1);

  switch (n_threads_log2)
  {
    case 9 /* 512 */: cuda::CudaRmsNormLine<T, 512><<< grid_size, block_size, smemSize >>>(d, x, w, eps, dim); break;
    case 8 /* 256 */: cuda::CudaRmsNormLine<T, 256><<< grid_size, block_size, smemSize >>>(d, x, w, eps, dim); break;
    case 7 /* 128 */: cuda::CudaRmsNormLine<T, 128><<< grid_size, block_size, smemSize >>>(d, x, w, eps, dim); break;
    case 6 /*  64 */: cuda::CudaRmsNormLine<T,  64><<< grid_size, block_size, smemSize >>>(d, x, w, eps, dim); break;
    case 5 /*  32 */: cuda::CudaRmsNormLine<T,  32><<< grid_size, block_size, smemSize >>>(d, x, w, eps, dim); break;
    case 4 /*  16 */: cuda::CudaRmsNormLine<T,  16><<< grid_size, block_size, smemSize >>>(d, x, w, eps, dim); break;
    case 3 /*   8 */: cuda::CudaRmsNormLine<T,   8><<< grid_size, block_size, smemSize >>>(d, x, w, eps, dim); break;
    case 2 /*   4 */: cuda::CudaRmsNormLine<T,   4><<< grid_size, block_size, smemSize >>>(d, x, w, eps, dim); break;
    case 1 /*   2 */: cuda::CudaRmsNormLine<T,   2><<< grid_size, block_size, smemSize >>>(d, x, w, eps, dim); break;
    case 0 /*   1 */: cuda::CudaRmsNormLine<T,   1><<< grid_size, block_size, smemSize >>>(d, x, w, eps, dim); break;
  }
#endif
}

#define FUNCTION(T) \
  template void RmsNormOperator<device::Cuda>::Eval<T>(T*, const T*, const T, size_t) const; \
  template void RmsNormOperator<device::Cuda>::Eval<T>(T*, const T*, const T, std::array<const size_t, 2>&&) const; \
  template void RmsNormOperator<device::Cuda>::Eval<T>(T*, const T*, const T*, const T, size_t) const; \
  template void RmsNormOperator<device::Cuda>::Eval<T>(T*, const T*, const T*, const T, std::array<const size_t, 2>&&) const;

#define TYPES  float

INSTANTIATE1(FUNCTION, (TYPES))

} // end of namespace grid
