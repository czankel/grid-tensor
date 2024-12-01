//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef TENSOR_SOURCE_CUDA_KERNELS_UTILS_H
#define TENSOR_SOURCE_CUDA_KERNELS_UTILS_H

namespace grid {
namespace cuda {

// FIXME: helper??
#if 0
  //if (tid == 0)
  printf("grid = (%i, %i, %i), block = (%i, %i, %i) bid (%i,%i,%i), tid (%i, %i, %i) gridSize: %d\n",
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    threadIdx.x, threadIdx.y, threadIdx.z,
    gridSize );
#endif

// The number of threads implemented in GPUs is currently fixed to 1024
static constexpr size_t MaxThreadCount = 1024;

// MakeDim3 is a helper function to create a dim3 value out of a span of n elements (n <=3)
template <typename T, size_t R>
inline dim3 MakeDim3(std::span<T, R> s)
{
  return [=] <std::size_t... I> (std::index_sequence<I...>) {
     return dim3{(static_cast<unsigned int>(s[I]))...};
  }(std::make_index_sequence<R>());
}

// GetSizes returns a tuple of the grid and block sizes for the provided size (single dimension)
// Block size is the number of threads in each thread block
// Grid size is the number of thread blocks in grid
inline auto GetSizes(size_t size)
{
  int block_size = std::min(size, MaxThreadCount);
  int grid_size = (int)ceil((float)size/block_size);
  return std::make_tuple(grid_size, block_size);
}

// FIXME: isn't first index row then col? so, returns {batch, col, row}
// FIXME: shouldn't be there a "minimum" somewhere??
// GetSizes returns a tuple of the 3-dimensional grid and block sizes for the provided n-dimensional sizes
template <size_t R, typename... S>
inline auto GetSizes(std::span<const size_t, R> dims, S... sizes)
{
  return [=] <std::size_t... I> (std::index_sequence<I...>, auto sz) {
     return std::make_tuple(
         dim3{((unsigned int)ceil((float)(dims[I])/std::min(static_cast<size_t>(std::get<I>(sz)), MaxThreadCount)))...},
         dim3{((unsigned int)std::min(static_cast<size_t>(std::get<I>(sz)), MaxThreadCount))...});
  }(std::make_index_sequence<R>(), std::make_tuple(sizes...));
}

} // end of namespace cuda
} // end of namespace grid

#endif  // TENSOR_SOURCE_CUDA_KERNELS_UTILS_H
