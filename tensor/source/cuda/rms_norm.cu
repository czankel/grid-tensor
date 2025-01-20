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

template <typename T, unsigned int BlockSize>
__device__ void CudaWarpReduce(volatile T* sdata, unsigned int tid, unsigned int dim)
{ // FIXME: tid vs tid_x for < dim??
  if (BlockSize >= 64 && tid + 32 < dim) sdata[tid] = sdata[tid] + sdata[tid + 32];
  if (BlockSize >= 32 && tid + 16 < dim) sdata[tid] = sdata[tid] + sdata[tid + 16];
  if (BlockSize >= 16 && tid +  8 < dim) sdata[tid] = sdata[tid] + sdata[tid +  8];
  if (BlockSize >=  8 && tid +  4 < dim) sdata[tid] = sdata[tid] + sdata[tid +  4];
  if (BlockSize >=  4 && tid +  2 < dim) sdata[tid] = sdata[tid] + sdata[tid +  2];
  if (BlockSize >=  2 && tid +  1 < dim) sdata[tid] = sdata[tid] + sdata[tid +  1];
}

template <typename T, unsigned int BlockSize>
__global__ void CudaRmsNorm(T* d, const T* x, const T eps, int dim)
{
  __shared__ T sdata[cuda::MaxThreadCount];

  unsigned int tid_x = threadIdx.x;
  unsigned int lane = threadIdx.y * dim;
  unsigned int tid = lane + tid_x;
  unsigned int idx_s = blockIdx.x * BlockSize + tid;
  unsigned int idx_e = lane + dim;
  unsigned int grid_size = gridDim.x * BlockSize;

  T sum{0};
  for (unsigned int i = idx_s; i < idx_e; i += grid_size)
    sum += x[i] * x[i];

  sdata[tid] = sum;

  __syncthreads();

  if (BlockSize >= 512) { if (tid_x < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (BlockSize >= 256) { if (tid_x < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (BlockSize >= 128) { if (tid_x < 64)  { sdata[tid] += sdata[tid + 64];  } __syncthreads(); }
  if (tid_x < 32) CudaWarpReduce<T, BlockSize>(sdata, tid, lane + dim);

  if (tid_x == 0)
    sdata[lane] = sqrt(sdata[lane] / dim + eps);

  __syncthreads();

  for (unsigned int i = idx_s; i < idx_e; i += grid_size)
    d[i] = x[i] / sdata[lane];
}

} // end of namespace cuda


template <typename T>
void RmsNormOperator<device::Cuda>::Eval(T* d, const T* x, const T eps, size_t rows, size_t cols) const
{
  const size_t dims[]{cols, rows};
  size_t warp_size = static_cast<size_t>(cuda::WarpSize); // FIXME:

  // use as many parallel threads (rows) as we can
  size_t blk_x = ((rows + warp_size - 1) / warp_size) * warp_size; // FIXME: why align up to warp_size??
  size_t blk_y = std::min(cuda::MaxThreadCount / blk_x, rows);
  blk_x = std::min(blk_x, cols);

  auto [grid_size, block_size] = cuda::GetSizes(std::span{dims}, blk_x, blk_y);

  size_t smem_size = cuda::MaxThreadCount;
  size_t n_threads = cols > 1024 ? 1024 : cols; // FIXME
  int n_threads_log2 = sizeof(n_threads) * 8 - __builtin_clzll(n_threads - 1);

  switch (n_threads_log2)
  {
    #define CUDA_RMS_NORM_CASE(BIT) \
      case BIT: cuda::CudaRmsNorm<T,1<<BIT><<<grid_size,block_size,smem_size>>>(d,x,eps,cols); break;
    INSTANTIATE1(CUDA_RMS_NORM_CASE, (9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
    default: throw std::runtime_error("invalid thread count");
  }
}

#define FUNCTION(T) \
  template void RmsNormOperator<device::Cuda>::Eval<T>(T*, const T*, const T, size_t, size_t) const; 

#define TYPES  float

INSTANTIATE1(FUNCTION, (TYPES))

} // end of namespace grid
