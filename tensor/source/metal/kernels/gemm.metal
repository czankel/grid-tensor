//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//


#include "../../instantiate.h"
#include "utils.h"

//
// Unoptimized (non-contiguous) matrix and vector multiplication.
//

template <typename T, typename U>
[[kernel]] void GeMMOperatorGeneric(device U* d,
                                    device const T* x,
                                    device const T* y,
                                    constant const size_t& dim_j,
                                    constant const size_t strides_d[2],
                                    constant const size_t strides_x[2],
                                    constant const size_t strides_y[2],
                                    uint2 dims [[thread_position_in_grid]])
{
  T sum{0};

  size_t idx_d = dims[0] * strides_d[0] + dims[1] * strides_d[1];
  size_t idx_x = dims[0] * strides_x[0];
  size_t idx_y = dims[1] * strides_y[1];

  for (uint j = 0; j < dim_j; j++)
    sum += x[idx_x + j * strides_x[1]] * y[idx_y + j * strides_y[0]];
  d[idx_d] = (T)sum;
}

#define FUNCTION(R, T) \
  template [[host_name(stringify(GeMMOperator ## R ## T))]]  \
  [[kernel]] void GeMMOperator ## R <T, T>( \
      device T*, \
      device const T*, \
      device const T*, \
      constant const size_t& dim_j, \
      constant const size_t strides_d[2], \
      constant const size_t strides_x[2], \
      constant const size_t strides_y[2], \
      uint2);

#define RANKS Generic
#define TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

INSTANTIATE2(FUNCTION, (RANKS), (TYPES))
