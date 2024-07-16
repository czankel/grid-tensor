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
[[kernel]] void RmsNormLine(const device T* x,
                            const device T*,
                            device T* y,
                            constant float& eps,
                            constant uint& dim,
                            constant uint&,
                            threadgroup float* local_inv_mean [[threadgroup(0)]],
                            threadgroup float* local_sums [[threadgroup(1)]],
                            uint gid [[threadgroup_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            __attribute__((unused)) uint line_size [[threads_per_threadgroup]],
                            uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  float acc = 0;
  x += gid * dim + lid * N_READS;

  if (lid * N_READS + N_READS <= dim)
  {
    for (int i = 0; i < N_READS; i++)
    {
      float xi = x[i];  // TODO: is there any perf difference to x[i]*x[i]?
      acc += xi * xi;
    }
  }
  else
  {
    for (int i = 0; i < N_READS; i++)
    {
      if ((lid * N_READS + i) < dim)
      {
        float xi = x[i];
        acc += xi * xi;
      }
    }
  }
  acc = metal::simd_sum(acc);

  if (simd_group_id == 0)
    local_sums[simd_lane_id] = 0;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_lane_id == 0)
    local_sums[simd_group_id] = acc;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_group_id == 0)
  {
    acc = metal::simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0)
      local_inv_mean[0] = metal::precise::rsqrt(acc / dim + eps);
  }

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  y += gid * dim + lid * N_READS;
  if (lid * N_READS + N_READS <= dim)
    for (int i = 0; i < N_READS; i++)
      y[i] = static_cast<T>(x[i] * local_inv_mean[0]);
  else
    for (int i = 0; i < N_READS; i++)
      if ((lid * N_READS + i) < dim)
        y[i] = static_cast<T>(x[i] * local_inv_mean[0]);
}


template <typename T, int N_READS = 4>
[[kernel]] void RmsNormLoop(const device T* x,
                            const device T*,
                            device T* y,
                            constant float& eps,
                            constant uint& dim,
                            constant uint&,
                            threadgroup float* local_inv_mean [[threadgroup(0)]],
                            threadgroup float* local_sums [[threadgroup(1)]],
                            uint gid [[threadgroup_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            uint line_size [[threads_per_threadgroup]],
                            uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  float acc = 0; // TODO: use T or float?

  x += gid * dim + lid * N_READS;

  for (uint r = 0; r < dim; r+= line_size * N_READS)
    if (r + lid * N_READS + N_READS <= dim)
      for (int i = 0; i < N_READS; i++)
      {
        float xi = x[i + r];
        acc += xi * xi;
      }
    else
      for (int i = 0; i < N_READS; i++)
        if ((r + lid * N_READS + i) < dim)
        {
          float xi = x[i + r];
          acc += xi * xi;
        }

  acc = metal::simd_sum(acc);

  if (simd_group_id == 0)
    local_sums[simd_lane_id] = 0;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_lane_id == 0)
    local_sums[simd_group_id] = acc;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_group_id == 0)
  {
    acc = metal::simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0)
      local_inv_mean[0] = metal::precise::rsqrt(acc / dim + eps);
  }

  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  y += gid * dim + lid + N_READS;
  for (uint r = 0; r < dim; r += line_size * N_READS)
    if (r + lid * N_READS + N_READS <= dim)
      for (int i = 0; i < N_READS; i++)
        y[r + i] = static_cast<T>(x[r + i] * local_inv_mean[0]);

    else
      for (int i = 0; i < N_READS; i++)
        if ((r + lid * N_READS + i) < dim)
          y[r + i] = static_cast<T>(x[r + i] * local_inv_mean[0]);
}


template <typename T, int N_READS = 4>	 // 4 per simd thread?
[[kernel]] void RmsNormWeightLine(const device T* x,
                                  const device T* W,
                                  device T* y,
                                  constant float& eps,
                                  constant uint& dim,
                                  constant uint& w_stride,
                                  threadgroup float* local_inv_mean [[threadgroup(0)]],
                                  threadgroup float* local_sums [[threadgroup(1)]],
                                  uint gid [[threadgroup_position_in_grid]],
                                  uint lid [[thread_position_in_threadgroup]],
                                  __attribute__((unused)) uint line_size [[threads_per_threadgroup]],
                                  uint simd_lane_id [[thread_index_in_simdgroup]],
                                  uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  float acc = 0;
  x += gid * dim + lid * N_READS;
  W += w_stride * lid * N_READS;

  if (lid * N_READS + N_READS <= dim)
  {
    for (int i = 0; i < N_READS; i++)
    {
      float xi = x[i];
      acc += xi * xi;
    }
  }
  else
  {
    for (int i = 0; i < N_READS; i++)
    {
      if ((lid * N_READS + i) < dim)
      {
        float xi = x[i];
        acc += xi * xi;
      }
    }
  }
  acc = metal::simd_sum(acc);

  if (simd_group_id == 0)
    local_sums[simd_lane_id] = 0;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_lane_id == 0)
    local_sums[simd_group_id] = acc;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_group_id == 0)
  {
    acc = metal::simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0)
      local_inv_mean[0] = metal::precise::rsqrt(acc / dim + eps);
  }

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  y += gid * dim + lid * N_READS;
  if (lid * N_READS + N_READS <= dim)
    for (int i = 0; i < N_READS; i++)
      y[i] = W[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
  else
    for (int i = 0; i < N_READS; i++)
      if ((lid * N_READS + i) < dim)
        y[i] = W[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
}


template <typename T, int N_READS = 4>
[[kernel]] void RmsNormWeightLoop(const device T* x,
                                  const device T* W,
                                  device T* y,
                                  constant float& eps,
                                  constant uint& dim,
                                  constant uint& w_stride,
                                  threadgroup float* local_inv_mean [[threadgroup(0)]],
                                  threadgroup float* local_sums [[threadgroup(1)]],
                                  uint gid [[threadgroup_position_in_grid]],
                                  uint lid [[thread_position_in_threadgroup]],
                                  uint line_size [[threads_per_threadgroup]],
                                  uint simd_lane_id [[thread_index_in_simdgroup]],
                                  uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  float acc = 0;

  x += gid * dim + lid * N_READS;
  W += w_stride * lid * N_READS;

  for (uint r = 0; r < dim; r+= line_size * N_READS)
    if (r + lid * N_READS + N_READS <= dim)
      for (int i = 0; i < N_READS; i++)
      {
        float xi = x[i + r];
        acc += xi * xi;
      }
    else
      for (int i = 0; i < N_READS; i++)
        if ((r + lid * N_READS + i) < dim)
        {
          float xi = x[i + r];
          acc += xi * xi;
        }

  acc = metal::simd_sum(acc);

  if (simd_group_id == 0)
    local_sums[simd_lane_id] = 0;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_lane_id == 0)
    local_sums[simd_group_id] = acc;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_group_id == 0)
  {
    acc = metal::simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0)
      local_inv_mean[0] = metal::precise::rsqrt(acc / dim + eps);
  }

  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  y += gid * dim + lid + N_READS;
  for (uint r = 0; r < dim; r += line_size * N_READS)
    if (r + lid * N_READS + N_READS <= dim)
      for (int i = 0; i < N_READS; i++)
        y[r + i] = W[w_stride * (i + r)] * static_cast<T>(x[r + i] * local_inv_mean[0]);

    else
      for (int i = 0; i < N_READS; i++)
        if ((r + lid * N_READS + i) < dim)
          y[r + i] = W[w_stride * (i + r)] * static_cast<T>(x[r + i] * local_inv_mean[0]);
}

#define RMS_NORM_OPS Line, Loop, WeightLine, WeightLoop
#define RMS_NORM_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

#define RMS_NORM_FUNCTION(O, T) \
  template [[host_name(stringify(RmsNorm ## O ## T))]]  \
  [[kernel]] void RmsNorm ## O<T>( \
    device const T*, device const T*, \
    device T*, \
    constant float&, \
    constant uint&, \
    constant uint&, \
    threadgroup float*, \
    threadgroup float*, \
    uint, uint, uint, uint, uint);

INSTANTIATE2(RMS_NORM_FUNCTION, (RMS_NORM_OPS), (RMS_NORM_TYPES))
