//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//


#include <grid/tensor/cuda/array.h>

namespace grid {

void CudaMalloc(void** ptr, size_t size)
{
  auto err = cudaMalloc(ptr, size);
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("cudaMalloc failed") + cudaGetErrorString(err));
}

void CudaFree(void* ptr)
{
  auto err = cudaFree(ptr);
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("cudaMalloc failed") + cudaGetErrorString(err));
}

} // end of namespace grid
