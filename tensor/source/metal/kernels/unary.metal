//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//


#include "../../instantiate.h"

#include "utils.h"

struct CopyOperator { template<typename T> inline T operator()(T x) { return x; } };
struct NegOperator  { template<typename T> inline T operator()(T x) { return -x; } };

//
// Fast unary opeator supporting scalars but without strides
//

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


#define FAST_FUNCTION(R, O, T) \
  template [[host_name(stringify(UnaryOperator ## R ## O ## T))]]  \
  [[kernel]] void UnaryOperator ## R <O ## Operator, T>(device const T*, device T*, uint);

#define FAST_RANKS S, V
#define FAST_OPS   Copy, Neg
#define FAST_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

INSTANTIATE3(FAST_FUNCTION, (FAST_RANKS), (FAST_OPS), (FAST_TYPES))

//
// Default unary opeator with source strides and matching rank
//

template <typename Op, typename T, typename U>
[[kernel]] void UnaryOperatorRank1(device const T* a,
                                   device U* c,
                                   constant const size_t& a_stride,
                                   uint pos [[thread_position_in_grid]])
{
  auto a_idx = metal::pos_to_index(pos, a_stride);
  c[pos] = Op()(a[a_idx]);
}

template <typename Op, typename T, typename U>
[[kernel]] void UnaryOperatorRank2(device const T* a,
                                   device U* c,
                                   constant const size_t a_strides[2],
                                   uint2 pos [[thread_position_in_grid]],
                                   uint2 grid_dim [[threads_per_grid]])
{
  auto a_idx = metal::pos_to_index(pos, a_strides);
  size_t c_idx = pos.x + (size_t)grid_dim.x * pos.y;
  c[c_idx] = Op()(a[a_idx]);
}

template <typename Op, typename T, typename U>
[[kernel]] void UnaryOperatorRank3(device const T* a,
                                   device U* c,
                                   constant const size_t a_strides[3],
                                   uint3 pos [[thread_position_in_grid]],
                                   uint3 grid_dim [[threads_per_grid]])
{
  auto a_idx = metal::pos_to_index(pos, a_strides);
  size_t c_idx = pos.x + (size_t)grid_dim.x * (pos.y + (size_t)grid_dim.y * pos.z);
  c[c_idx] = Op()(a[a_idx]);
}


#define FULL_OPS   Copy, Neg
#define FULL_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

#define RANK1_FUNCTION(O, T) \
  template [[host_name(stringify(UnaryOperatorRank1 ## O ## T))]]  \
  [[kernel]] void UnaryOperatorRank1<O ## Operator, T, T>( \
    device const T*, device T*, \
    constant const size_t&, uint);

INSTANTIATE2(RANK1_FUNCTION, (FULL_OPS), (FULL_TYPES))

#define RANK2_FUNCTION(O, T) \
  template [[host_name(stringify(UnaryOperatorRank2 ## O ## T))]]  \
  [[kernel]] void UnaryOperatorRank2<O ## Operator, T, T>( \
    device const T*, device T*, \
    constant const size_t[2], uint2, uint2);

INSTANTIATE2(RANK2_FUNCTION, (FULL_OPS), (FULL_TYPES))

#define RANK3_FUNCTION(O, T) \
  template [[host_name(stringify(UnaryOperatorRank3 ## O ## T))]]  \
  [[kernel]] void UnaryOperatorRank3<O ## Operator, T, T>( \
    device const T*, device T*, \
    constant const size_t[3], uint3, uint3);

INSTANTIATE2(RANK3_FUNCTION, (FULL_OPS), (FULL_TYPES))
