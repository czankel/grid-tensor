//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//


#include <grid/tensor//tensor.h>
#include <grid/tensor/cuda/binary.h>
#include <grid/tensor/cuda/device.h>

#include "../utils.h"


//
// Binary Operators
//

template <> struct grid::AddOperator<grid::device::Cuda>
{ template<typename T> inline __device__ T operator()(T x, T y) { return x + y; } };
template <> struct grid::SubOperator<grid::device::Cuda>
{ template<typename T> inline __device__ T operator()(T x, T y) { return x - y; } };
template <> struct grid::MulOperator<grid::device::Cuda>
{ template<typename T> inline __device__ T operator()(T x, T y) { return x * y; } };
template <> struct grid::DivOperator<grid::device::Cuda>
{ template<typename T> inline __device__ T operator()(T x, T y) { return x / y; } };

// threadIdx

template <template <typename> typename TOperator, typename T>
__global__ void CudaBinarySS(T* c, const T* a, const T* b)
{
  c[0] = TOperator<grid::device::Cuda>()(a[0], b[0]);
}

template <template <typename> typename TOperator, typename T>
__global__ void CudaBinaryVS(T* c, const T* a, const T* b)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  c[index] = TOperator<grid::device::Cuda>()(a[index], b[0]);
}

template <template <typename> typename TOperator, typename T>
__global__ void CudaBinarySV(T* c, const T* a, const T* b)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  if (index < n)
    c[index] = TOperator<grid::device::Cuda>()(a[0], b[index]);
}

template <template <typename> typename TOperator, typename T>
__global__ void CudaBinaryVV(T* c, const T* a, const T* b)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  c[index] = TOperator<grid::device::Cuda>()(a[index], b[index]);
}


template <template <typename> typename TOperator>
template <typename T>
void grid::BinaryOperator<TOperator<grid::device::Cuda>>::eval(
    T* c, const T* a, const T* b,
    size_t folded_dimensions) const
{
  printf("BinaryOperator 3\n");
  // Number of threads in each thread block FIXME
  int blockSize = 1024;
       
  // Number of thread blocks in grid
  int gridSize = (int)ceil((float)folded_dimensions/blockSize);
           
  CudaBinaryVV<TOperator, T><<<gridSize, blockSize>>>(c, a, b);
}

template <template <typename> typename TOperator>
template <typename T>
void grid::BinaryOperator<TOperator<grid::device::Cuda>>::eval(
    T* c, const T* a, const T* b,
    size_t rank, const size_t* dimensions,
    const ssize_t* strides0, const ssize_t* strides1, const ssize_t* strides2) const
{
  printf("BinaryOperator 1\n");
  //CudaBinaryEval(..)
}

template <template <typename> typename TOperator>
template <typename T>
void grid::BinaryOperator<TOperator<grid::device::Cuda>>::eval(
    T* c, const T* a, const T* b,
    size_t folded_dimensions,
    size_t rank, const size_t* dimensions, const ssize_t* strides0, const ssize_t* strides1, const ssize_t* strides2) const
{
#if 0
  // FIXME: constexpr and move to cuda/binary.h?
  if (strides0[rank-1] == 0)
  {
    if (strides1[rank-1] == 0 || strides2[rank-1] == 0)
      CudaBinaryOperatorSS();
    else
      throw ;
  }
  else if (strides1[rank - 1 ] == 0)
    CudaBinaryOperatorSV();
  else if (strides2[rank - 1] == 0)
    CudaBinaryOperatorSV();
  else
    CudaBinaryOperatorVV();
#endif

  printf("BinaryOperator 2\n");
  //CudaBinaryEval(..)
}


//BinaryOperator<Add<device::Cuda>, float>;

#define INSTANCE_TEMPLATE(O, T) \
  grid::BinaryOperator<grid::O ##Operator<grid::device::Cuda>> _g_Cuda_Binary_##Operator_##T; \
  template void grid::BinaryOperator<grid::O ##Operator<grid::device::Cuda>>::eval< T >( \
      T*, const T*, const T*, size_t) const; \
  template void grid::BinaryOperator<grid::O ##Operator<grid::device::Cuda>>::eval< T >( \
      T*, const T*, const T*, size_t, const size_t*, const ssize_t*, const ssize_t*, const ssize_t*) const; \
  template void grid::BinaryOperator<grid::O ##Operator<grid::device::Cuda>>::eval<T>( \
      T*, const T*, const T*, size_t, const size_t, const size_t*, const ssize_t*, const ssize_t*, const ssize_t*) const;

#define FUNCTION_OPS    Add //, Sub, Mul, Div
#define FUNCTION_TYPES  double,int

INSTANTIATE2(INSTANCE_TEMPLATE, (FUNCTION_OPS), (FUNCTION_TYPES))
