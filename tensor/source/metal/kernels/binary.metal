//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//


#include "../../instantiate.h"

#include "utils.h"

struct AddOperator { template<typename T> inline T operator()(T x, T y) { return x + y; } };
struct SubOperator { template<typename T> inline T operator()(T x, T y) { return x - y; } };
struct MulOperator { template<typename T> inline T operator()(T x, T y) { return x * y; } };
struct DivOperator { template<typename T> inline T operator()(T x, T y) { return x / y; } };

//
// Fast binary opeator supporting scalars but without strides
//

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorSS(device U* d,
                                 device const T* x,
                                 device const T* y,
                                 uint index [[thread_position_in_grid]])
{
  d[index] = Op()(x[0], y[0]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorSV(device U* d,
                                 device const T* x,
                                 device const T* y,
                                 uint index [[thread_position_in_grid]])
{
  d[index] = Op()(x[0], y[index]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorVS(device U* d,
                                 device const T* x,
                                 device const T* y,
                                 uint index [[thread_position_in_grid]])
{
  d[index] = Op()(x[index], y[0]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorVV(device U* d,
                                 device const T* x,
                                 device const T* y,
                                 uint index [[thread_position_in_grid]])
{
  d[index] = Op()(x[index], y[index]);
}

#define FAST_FUNCTION(R, O, T) \
  template [[host_name(stringify(BinaryOperator ## R ## O ## T))]]  \
  [[kernel]] void BinaryOperator ## R <O ## Operator, T, T>(device T*, device const T*, device const T*, uint);

#define FAST_RANKS SS, SV, VS, VV
#define FAST_OPS   Add, Sub, Mul, Div
#define FAST_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

INSTANTIATE3(FAST_FUNCTION, (FAST_RANKS), (FAST_OPS), (FAST_TYPES))

//
// Default binary opeator with source strides and matching rank
//

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorRank1(device U* d,
                                    device const T* x,
                                    device const T* y,
                                    constant const size_t& stride_x,
                                    constant const size_t& stride_y,
                                    uint pos [[thread_position_in_grid]])
{
  auto idx_x = metal::pos_to_index(pos, stride_x);
  auto idx_y = metal::pos_to_index(pos, stride_y);
  d[pos] = Op()(x[idx_x], y[idx_y]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorRank2(device U* d,
                                    device const T* x,
                                    device const T* y,
                                    constant const size_t strides_x[2],
                                    constant const size_t strides_y[2],
                                    uint2 pos [[thread_position_in_grid]],
                                    uint2 grid_dim [[threads_per_grid]])
{
  auto idx_x = metal::pos_to_index(pos, strides_x);
  auto idx_y = metal::pos_to_index(pos, strides_y);
  size_t c_idx = pos.x + (size_t)grid_dim.x * pos.y;
  d[c_idx] = Op()(x[idx_x], y[idx_y]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorRank3(device U* d,
                                    device const T* x,
                                    device const T* y,
                                    constant const size_t strides_x[3],
                                    constant const size_t strides_y[3],
                                    uint3 pos [[thread_position_in_grid]],
                                    uint3 grid_dim [[threads_per_grid]])
{
  auto idx_x = metal::pos_to_index(pos, strides_x);
  auto idx_y = metal::pos_to_index(pos, strides_y);
  size_t c_idx = pos.x + (size_t)grid_dim.x * (pos.y + (size_t)grid_dim.y * pos.z);
  d[c_idx] = Op()(x[idx_x], y[idx_y]);
}


#define FULL_OPS   Add, Sub, Mul, Div
#define FULL_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

#define RANK1_FUNCTION(O, T) \
  template [[host_name(stringify(BinaryOperatorRank1 ## O ## T))]]  \
  [[kernel]] void BinaryOperatorRank1<O ## Operator, T, T>( \
    device T*, device const T*, device const T*, \
    constant const size_t&, constant const size_t&, uint);

INSTANTIATE2(RANK1_FUNCTION, (FULL_OPS), (FULL_TYPES))

#define RANK2_FUNCTION(O, T) \
  template [[host_name(stringify(BinaryOperatorRank2 ## O ## T))]]  \
  [[kernel]] void BinaryOperatorRank2<O ## Operator, T, T>( \
    device T*, device const T*, const device T*, \
    constant const size_t[2], constant const size_t[2], uint2, uint2);

INSTANTIATE2(RANK2_FUNCTION, (FULL_OPS), (FULL_TYPES))

#define RANK3_FUNCTION(O, T) \
  template [[host_name(stringify(BinaryOperatorRank3 ## O ## T))]]  \
  [[kernel]] void BinaryOperatorRank3<O ## Operator, T, T>( \
    device T*, device const T*, device const T*, \
    constant const size_t[3], constant const size_t[3], uint3, uint3);

INSTANTIATE2(RANK3_FUNCTION, (FULL_OPS), (FULL_TYPES))
