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
[[kernel]] void BinaryOperatorSS(device const T* a,
                                 device const T* b,
                                 device U* d,
                                 uint index [[thread_position_in_grid]])
{
  d[index] = Op()(a[0], b[0]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorSV(device const T* a,
                                 device const T* b,
                                 device U* d,
                                 uint index [[thread_position_in_grid]])
{
  d[index] = Op()(a[0], b[index]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorVS(device const T* a,
                                 device const T* b,
                                 device U* d,
                                 uint index [[thread_position_in_grid]])
{
  d[index] = Op()(a[index], b[0]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorVV(device const T* a,
                                 device const T* b,
                                 device U* d,
                                 uint index [[thread_position_in_grid]])
{
  d[index] = Op()(a[index], b[index]);
}

#define FAST_FUNCTION(R, O, T) \
  template [[host_name(stringify(BinaryOperator ## R ## O ## T))]]  \
  [[kernel]] void BinaryOperator ## R <O ## Operator, T, T>(device const T*, device const T*, device T*, uint);

#define FAST_RANKS SS, SV, VS, VV
#define FAST_OPS   Add, Sub, Mul, Div
#define FAST_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

INSTANTIATE3(FAST_FUNCTION, (FAST_RANKS), (FAST_OPS), (FAST_TYPES))

//
// Default binary opeator with source strides and matching rank
//

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorRank1(device const T* a,
                                    device const T* b,
                                    device U* d,
                                    constant const size_t& a_stride,
                                    constant const size_t& b_stride,
                                    uint pos [[thread_position_in_grid]])
{
  auto a_idx = metal::pos_to_index(pos, a_stride);
  auto b_idx = metal::pos_to_index(pos, b_stride);
  d[pos] = Op()(a[a_idx], b[b_idx]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorRank2(device const T* a,
                                    device const T* b,
                                    device U* d,
                                    constant const size_t a_strides[2],
                                    constant const size_t b_strides[2],
                                    uint2 pos [[thread_position_in_grid]],
                                    uint2 grid_dim [[threads_per_grid]])
{
  auto a_idx = metal::pos_to_index(pos, a_strides);
  auto b_idx = metal::pos_to_index(pos, b_strides);
  size_t c_idx = pos.x + (size_t)grid_dim.x * pos.y;
  d[c_idx] = Op()(a[a_idx], b[b_idx]);
}

template <typename Op, typename T, typename U>
[[kernel]] void BinaryOperatorRank3(device const T* a,
                                    device const T* b,
                                    device U* d,
                                    constant const size_t a_strides[3],
                                    constant const size_t b_strides[3],
                                    uint3 pos [[thread_position_in_grid]],
                                    uint3 grid_dim [[threads_per_grid]])
{
  auto a_idx = metal::pos_to_index(pos, a_strides);
  auto b_idx = metal::pos_to_index(pos, b_strides);
  size_t c_idx = pos.x + (size_t)grid_dim.x * (pos.y + (size_t)grid_dim.y * pos.z);
  d[c_idx] = Op()(a[a_idx], b[b_idx]);
}


#define FULL_OPS   Add, Sub, Mul, Div
#define FULL_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

#define RANK1_FUNCTION(O, T) \
  template [[host_name(stringify(BinaryOperatorRank1 ## O ## T))]]  \
  [[kernel]] void BinaryOperatorRank1<O ## Operator, T, T>( \
    device const T*, device const T*, device T*, \
    constant const size_t&, constant const size_t&, uint);

INSTANTIATE2(RANK1_FUNCTION, (FULL_OPS), (FULL_TYPES))

#define RANK2_FUNCTION(O, T) \
  template [[host_name(stringify(BinaryOperatorRank2 ## O ## T))]]  \
  [[kernel]] void BinaryOperatorRank2<O ## Operator, T, T>( \
    device const T*, device const T*, device T*, \
    constant const size_t[2], constant const size_t[2], uint2, uint2);

INSTANTIATE2(RANK2_FUNCTION, (FULL_OPS), (FULL_TYPES))

#define RANK3_FUNCTION(O, T) \
  template [[host_name(stringify(BinaryOperatorRank3 ## O ## T))]]  \
  [[kernel]] void BinaryOperatorRank3<O ## Operator, T, T>( \
    device const T*, device const T*, device T*, \
    constant const size_t[3], constant const size_t[3], uint3, uint3);

INSTANTIATE2(RANK3_FUNCTION, (FULL_OPS), (FULL_TYPES))
