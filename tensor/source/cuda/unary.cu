//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//


#include <grid/tensor//tensor.h>
#include <grid/tensor/cuda/device.h>
#include <grid/tensor/cuda/unary.h>

#include "../instantiate.h"


//
// Unary Operators
//

template <> struct grid::CopyOperator<grid::device::Cuda>
{ template<typename T> inline __device__ T operator()(T x) { return x; } };
template <> struct grid::NegOperator<grid::device::Cuda>
{ template<typename T> inline __device__ T operator()(T x) { return -x; } };


template <template <typename> typename TOperator, typename T>
__global__ void CudaUnaryVector(T* c, const T* a)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  c[index] = TOperator<grid::device::Cuda>()(a[index]);
}

// Two versions actually requires (scalar and vector)
template <template <typename> typename TOperator, typename T>
__global__ void CudaUnaryScalar(T* c, const T* a)
{
  c[0] = TOperator<grid::device::Cuda>()(a[0]);
}


template <template <typename> typename TOperator>
template <typename T>
void grid::UnaryOperator<TOperator<grid::device::Cuda>>::eval(T* c, const T* a, size_t folded_dimensions) const
{
  // Number of threads in each thread block FIXME
  int blockSize = 1024;

  // Number of thread blocks in grid
  int gridSize = (int)ceil((float)folded_dimensions/blockSize);

  CudaUnaryV<TOperator, T><<<gridSize, blockSize>>>(c, a, folded_dimensions);
}

template <template <typename> typename TOperator>
template <typename T>
void grid::UnaryOperator<TOperator<grid::device::Cuda>>::eval(
    T* c, const T* a, const T* b,
    size_t rank, const size_t* dimensions,
    const ssize_t* strides0, const ssize_t* strides1, const ssize_t* strides2) const
{
  printf("UnaryOperator 1\n");
  //CudaUnaryEval(..)
}

template <template <typename> typename TOperator>
template <typename T>
void grid::UnaryOperator<TOperator<grid::device::Cuda>>::eval(
    T* c, const T* a, const T* b,
    size_t folded_dimensions,
    size_t rank, const size_t* dimensions, const ssize_t* strides0, const ssize_t* strides1, const ssize_t* strides2) const
{
#if 0
  // FIXME: constexpr and move to cuda/binary.h?
  if (strides0[rank-1] == 0)
  {
    if (strides1[rank-1] == 0 || strides2[rank-1] == 0)
      CudaUnaryOperatorSS();
    else
      throw ;
  }
  else if (strides1[rank - 1 ] == 0)
    CudaUnaryOperatorSV();
  else if (strides2[rank - 1] == 0)
    CudaUnaryOperatorSV();
  else
    CudaUnaryOperatorVV();
#endif

  printf("UnaryOperator 2\n");
  //CudaUnaryEval(..)
}


//UnaryOperator<Add<device::Cuda>, float>;

#define FUNCTION(O, T) \
  grid::UnaryOperator<grid::O ##Operator<grid::device::Cuda>> _g_Cuda_Unary_##Operator_##T; \
  template void grid::UnaryOperator<grid::O ##Operator<grid::device::Cuda>>::eval< T >( \
      T*, const T*, const T*, size_t) const; \
  template void grid::UnaryOperator<grid::O ##Operator<grid::device::Cuda>>::eval< T >( \
      T*, const T*, const T*, size_t, const size_t*, const ssize_t*, const ssize_t*, const ssize_t*) const; \
  template void grid::UnaryOperator<grid::O ##Operator<grid::device::Cuda>>::eval<T>( \
      T*, const T*, const T*, size_t, const size_t, const size_t*, const ssize_t*, const ssize_t*, const ssize_t*) const;

#define OPS    Copy, Neg
#define TYPES  double,int

INSTANTIATE2(FUNCTION, (OPS), (TYPES))
