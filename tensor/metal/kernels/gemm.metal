//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//


#include "utils.h"

// Fast...
template <typename T, typename U>
[[kernel]] void GemmOperatorMM(device const T* a,
                               device const T* b,
                               device U* c,
                               constant const size_t& dim_j,
                               constant const size_t strides0[2],
                               constant const size_t strides1[2],
                               constant const size_t strides2[2],
                               uint2 dims [[thread_position_in_grid]])
{
  T sum{0};

  size_t index_a = dims[0] * strides1[0];
  size_t index_b = dims[1];
  size_t index_c = dims[0] * strides0[0] + dims[1];

  for (uint j = 0; j < dim_j; j++)
    sum += a[index_a + j] * b[index_b + j * strides2[0]];
  c[index_c] = (T)sum;
}

#define FUNCTION_TEMPLATE(R, T) \
  template [[host_name(stringify(GemmOperator ## R ## T))]]  \
  [[kernel]] void GemmOperator ## R <T, T>( \
	      device const T*, \
        device const T*, \
        device T*, \
        constant const size_t& dim_j, \
        constant const size_t strides0[2], \
        constant const size_t strides1[2], \
        constant const size_t strides2[2], \
 	      uint2);
 
#define ITERATE_RANKS MM

#define ITERATE_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

INSTANTIATE2(FUNCTION_TEMPLATE, (ITERATE_RANKS), (ITERATE_TYPES))
