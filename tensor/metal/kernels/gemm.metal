//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//


#include "utils.h"

// Fast...
template <typename Op, typename T, typename U>
[[kernel]] void GemmOperatorMM(device const T* a,
                               device const T* b,
                               device U* c,
                               constant const size_t dims[3],
                               constant const size_t strides0[2],
                               constant const size_t strides1[2],
                               constant const size_t strides2[2],
                               uint index [[thread_position_in_grid]])
{
  for (size_t m = 0; m < dims[0]; m++)
  {
    T* c1 = c;
    const T* b1 = b;
    for (size_t n = 0; n < dims[2]; n++)
    {
      const T* a2 = a;
      const T* b2 = b1;
      T sum{0};
      for (uint k = 0; k < dims[1]; k++)
      {
         sum += *a2 * *b2;
         a2 += strides1[1];
         b2 += strides2[0];
      }
      *c1 =sum;
      c1 += strides0[1];
      b1 += strides2[1];
    }
    c += strides0[0];
    a += strides1[0];
  }
}

#define FUNCTION_TEMPLATE(R, O, T) \
  template [[host_name(stringify(GemmOperator ## R ## O ## T))]]  \
  [[kernel]] void GemmOperator ## R <O ## Operator, T, T>(device const T*, device const T*, device T*, uint);

#define ITERATE_RANKS MM
#define ITERATE_OPS   Slow
#define ITERATE_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

INSTANTIATE_FUNCTIONS(FUNCTION_TEMPLATE, (ITERATE_RANKS), (ITERATE_OPS), (ITERATE_TYPES))
