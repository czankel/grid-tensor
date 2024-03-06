//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//


#include "utils.h"

struct CopyOperator { template<typename T> inline T operator()(T x) { return x; } };
struct NegOperator  { template<typename T> inline T operator()(T x) { return -x; } };


template <typename Op, typename T, typename U>
[[kernel]] void UnaryOperatorS(device const T* a,
                               device U* c,
                               uint index [[thread_position_in_grid]])
{
  c[index] = Op()(a[index]);
}
template <typename Op, typename T, typename U>
[[kernel]] void UnaryOperatorV(device const T* a,
                               device U* c,
                               uint index [[thread_position_in_grid]])
{
  c[index] = Op()(a[index]);
}


#define FUNCTION_TEMPLATE(R, O, T) \
  template [[host_name(stringify(UnaryOperator ## R ## O ## T))]]  \
  [[kernel]] void UnaryOperator ## R <O ## Operator, T>(device const T*, device T*, uint);

#define FUNCTION_RANKS S, V
#define FUNCTION_OPS   Copy, Neg
#define FUNCTION_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat 

INSTANTIATE_FUNCTIONS(FUNCTION_TEMPLATE, (FUNCTION_RANKS), (FUNCTION_OPS), (FUNCTION_TYPES))
