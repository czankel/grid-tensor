//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//


#include "utils.h"

struct AddOperator { template<typename T> inline T operator()(T x, T y) { return x + y; } };
struct SubOperator { template<typename T> inline T operator()(T x, T y) { return x - y; } };
struct MulOperator { template<typename T> inline T operator()(T x, T y) { return x * y; } };
struct DivOperator { template<typename T> inline T operator()(T x, T y) { return x / y; } };


template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorSS(device const T* a,
                                 device const T* b,
                                 device U* c,
                                 uint index [[thread_position_in_grid]])
{
  c[index] = Op()(a[0], b[0]);
}
template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorSV(device const T* a,
                                 device const T* b,
                                 device U* c,
                                 uint index [[thread_position_in_grid]])
{
  c[index] = Op()(a[0], b[index]);
}
template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorVS(device const T* a,
                                 device const T* b,
                                 device U* c,
                                 uint index [[thread_position_in_grid]])
{
  c[index] = Op()(a[index], b[0]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorVV(device const T* a,
                                 device const T* b,
                                 device U* c,
                                 uint index [[thread_position_in_grid]])
{
  c[index] = Op()(a[index], b[index]);
}


#define FUNCTION_TEMPLATE(R, O, T) \
  template [[host_name(stringify(BinaryOperator ## R ## O ## T))]]  \
  [[kernel]] void BinaryOperator ## R <O ## Operator, T, T>(device const T*, device const T*, device T*, uint);

#define FUNCTION_RANKS SS, SV, VS, VV
#define FUNCTION_OPS   Add, Sub, Mul, Div
#define FUNCTION_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat 

INSTANTIATE3(FUNCTION_TEMPLATE, (FUNCTION_RANKS), (FUNCTION_OPS), (FUNCTION_TYPES))
