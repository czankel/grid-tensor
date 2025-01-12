//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "../../instantiate.h"

#include "utils.h"

template <typename T, int N_READS = 4>	 // 4 per simd thread?
[[kernel]] void SoftMaxLine(device T* d,
                            const device T* x,
                            constant float& eps,
                            constant uint& dim,
                            threadgroup float* local_max [[threadgroup(1)]],
                            threadgroup float* local_sum [[threadgroup(2)]],
                            uint gid [[threadgroup_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            __attribute__((unused)) uint line_size [[threads_per_threadgroup]],
                            uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  T max_value{};
  x += gid * dim + lid * N_READS;

  if (lid * N_READS + N_READS <= dim)
    for (int i = 0; i < N_READS; i++)
      max_value = metal::max(max_value, x[i]);
  else
    for (int i = 0; i < N_READS; i++)
      if ((lid * N_READS + i) < dim)
        max_value = metal::max(max_value, x[i]);

  max_value = metal::simd_max(max_value);

  if (simd_group_id == 0)
    local_max[simd_lane_id] = 0;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_lane_id == 0)
    local_max[simd_group_id] = max_value;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  T sum{};

  d += gid * dim + lid * N_READS;
  if (lid * N_READS + N_READS <= dim)
    for (int i = 0; i < N_READS; i++)
    {
      d[i] = static_cast<T>(metal::precise::exp(x[i] - local_max[0]));
      sum += d[i];
    }
  else
    for (int i = 0; i < N_READS; i++)
      if ((lid * N_READS + i) < dim)
      {
        d[i] = static_cast<T>(metal::precise::exp(x[i] - local_max[0]));
        sum += d[i];
      }

  sum = metal::simd_sum(sum);

  if (simd_group_id == 0)
    local_sum[simd_lane_id] = 0;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_group_id == 0)
    local_sum[simd_group_id] = sum;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  T scalar = sum + eps;

  if (lid * N_READS + N_READS <= dim)
    for (int i = 0; i < N_READS; i++)
      d[i] = d[i] / scalar;
  else
    for (int i = 0; i < N_READS; i++)
      if ((lid * N_READS + i) < dim)
        d[i] = d[i] / scalar;
}

template <typename T, int N_READS = 4>	 // 4 per simd thread?
[[kernel]] void SoftMaxLoop(device T* d,
                            device const T* x,
                            constant float& eps,
                            constant uint& dim,
                            threadgroup float* local_max [[threadgroup(1)]],
                            threadgroup float* local_sum [[threadgroup(2)]],
                            uint gid [[threadgroup_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            __attribute__((unused)) uint line_size [[threads_per_threadgroup]],
                            uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  T max_value = 0;
  x += gid * dim + lid * N_READS;

  for (uint r = 0; r < dim; r+= line_size * N_READS)
    if (r + lid * N_READS + N_READS <= dim)
      for (int i = 0; i < N_READS; i++)
        max_value = metal::max(max_value, x[i]);
    else
      for (int i = 0; i < N_READS; i++)
        if (r + lid * N_READS + i < dim)
          max_value = metal::max(max_value, x[i]);

  max_value = metal::simd_max(max_value);

  if (simd_group_id == 0)
    local_max[simd_lane_id] = 0;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_lane_id == 0)
    local_max[simd_group_id] = max_value;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  T sum{};

  d += gid * dim + lid * N_READS;
  if (lid * N_READS + N_READS <= dim)
    for (int i = 0; i < N_READS; i++)
    {
      d[i] = static_cast<T>(metal::precise::exp(x[i] - local_max[0]));
      sum += d[i];
    }
  else
    for (int i = 0; i < N_READS; i++)
      if ((lid * N_READS + i) < dim)
      {
        d[i] = static_cast<T>(metal::precise::exp(x[i] - local_max[0]));
        sum += d[i];
      }

  sum = metal::simd_sum(sum);

  if (simd_group_id == 0)
    local_sum[simd_lane_id] = 0;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_group_id == 0)
    local_sum[simd_group_id] = sum;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  T scalar = sum + eps;

  d += gid * dim + lid * N_READS;
  for (uint r = 0; r < dim; r += line_size * N_READS)
    if (r + lid * N_READS + N_READS <= dim)
      for (int i = 0; i < N_READS; i++)
        d[i] = d[i] / scalar;
    else
      for (int i = 0; i < N_READS; i++)
        if (r + lid * N_READS + i < dim)
          d[i] = d[i] / scalar;
}


#define SOFT_MAX_OPS Line, Loop
#define SOFT_MAX_TYPES uint8_t, uint16_t, uint32_t, /*uint64_t,*/ int8_t, int16_t, int32_t, /*int64_t,*/ half, float // FIXME , bfloat

#define SOFT_MAX_FUNCTION(O, T) \
  template [[host_name(stringify(SoftMax ## O ## T))]]  \
  [[kernel]] void SoftMax ## O<T>( \
    device T*, \
    device const T*, \
    constant float&, \
    constant uint&, \
    threadgroup float*, \
    threadgroup float*, \
    uint, uint, uint, uint, uint);

INSTANTIATE2(SOFT_MAX_FUNCTION, (SOFT_MAX_OPS), (SOFT_MAX_TYPES))
