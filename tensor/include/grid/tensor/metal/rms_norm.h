//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_METAL_RMS_NORM_H
#define GRID_TENSOR_METAL_RMS_NORM_H

#include <math.h>
#include <tuple>

#include "binary.h"

namespace grid {

namespace {
  template <typename> struct Eps {};
  template <> struct Eps<float>  { constexpr static float  default_value = 1e-5f; float  value; };
  template <> struct Eps<double> { constexpr static double default_value = 1e-5f; double value; };
}

template <> class RmsNormOperator<device::Metal>
{
  // TODO: assumes "x" and "w" are contiguous
  template <typename T>
  void eval(MTL::Buffer* d, const MTL::Buffer* x, auto dimensions, auto strides_d, auto strides_x) const
  {
    // TODO: define these at a more centralized location, are these defined in any metal headers?
    const int line_limit = 4096;
    const int simd_size = 32;
    const int n_reads = 4;

    auto& device = device::Metal::GetDevice();
    auto& encoder = device.Encoder();

    int row_size = dimensions.back();
    int n_rows = std::accumulate(std::begin(dimensions), std::end(dimensions) - 1, 1, std::multiplies<size_t>());

    MTL::ComputePipelineState* pipeline;
    size_t threadgroup_size;

    if (row_size > line_limit)
    {
      static metal::Kernel<T> kernel("RmsNormLoop");
      pipeline = kernel.ComputePipelineState();
      threadgroup_size = (((row_size + n_reads - 1) & -n_reads) + simd_size) & -simd_size;
    }
    else
    {
      static metal::Kernel<T> kernel("RmsNormLine");
      pipeline = kernel.ComputePipelineState();
      threadgroup_size = pipeline->maxTotalThreadsPerThreadgroup();
    }

    encoder->setComputePipelineState(pipeline);

    MTL::Size grid_size = MTL::Size(n_rows * threadgroup_size, 1, 1);
    MTL::Size group_size = MTL::Size(threadgroup_size, 1, 1);

    T eps = Eps<T>::default_value;

    encoder->setBuffer(x, 0, 0);
    // 1 is unused
    encoder->setBuffer(d, 0, 2);
    encoder->setBytes(&eps, sizeof(eps), 3);
    encoder->setBytes(&row_size, sizeof(int), 4);
    // 5 is unused

    // TODO: hard-coded to float?
    encoder->setThreadgroupMemoryLength(simd_size * sizeof(float), 1);
    encoder.DispatchThreads(grid_size, group_size);

    device.Wait();
  }


  template <typename T>
  void eval(MTL::Buffer* d, const MTL::Buffer* x, const MTL::Buffer* w,
            auto dimensions, auto strides_d, auto strides_x, auto strides_w) const
  {
    const int simd_size = 32;
    const int line_limit = 4096;
    const int n_reads = 4;

    auto& device = device::Metal::GetDevice();
    auto& encoder = device.Encoder();


    int row_size = dimensions.back();
    int n_rows = std::accumulate(std::begin(dimensions), std::end(dimensions) - 1, std::multiplies<int>());

    MTL::ComputePipelineState* pipeline;
    size_t threadgroup_size;

    if (row_size > line_limit)
    {
      static metal::Kernel<T> kernel("RmsNormLoop");
      pipeline = kernel.ComputePipelineState();
      threadgroup_size = (((row_size + n_reads - 1) & -n_reads) + simd_size) & -simd_size;
    }
    else
    {
      static metal::Kernel<T> kernel("RmsNormLine");
      pipeline = kernel.ComputePipelineState();
      threadgroup_size = pipeline->maxTotalThreadsPerThreadgroup();
    }

    MTL::Size grid_size = MTL::Size(n_rows * threadgroup_size, 1, 1);
    MTL::Size group_size = MTL::Size(threadgroup_size, 1, 1);

    uint32_t w_stride = strides_w.back();
    T eps = Eps<T>::default_value;

    encoder->setComputePipelineState(pipeline);

    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(w, 0, 1);
    encoder->setBuffer(d, 0, 2);
    encoder->setBytes(&eps, sizeof(eps), 3);
    encoder->setBytes(&row_size, sizeof(int), 4);
    encoder->setBytes(&w_stride, sizeof(uint32_t), 5);
    encoder->setThreadgroupMemoryLength(simd_size * sizeof(float), 1);
    encoder.DispatchThreads(grid_size, group_size);

    device.Wait();
  }

 public:

  // d = RMS(x) * x  -- Note: requires contiguous (not asserted)
  template<std::ranges::input_range I, std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);

    std::span strides_d(first_d.Strides());
    std::span strides_x(first_x.Strides());

    eval<value_type>(first_d.Buffer(), first_x.Buffer(), first_x.Extents(), strides_d, strides_x);
  }


  // d = IMS(x) * x @ W  -- Note: required contiguous (not asserted)
  template<std::ranges::input_range I1, std::ranges::input_range I2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I1>, std::ranges::iterator_t<O>>
  void operator()(I1&& in1, I2&& in2, O&& out) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first_r = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in1);
    auto first_W = std::ranges::cbegin(in2);

    std::span strides_r(first_r.Strides());
    std::span strides_x(first_x.Strides());
    std::span strides_W(first_W.Strides());

    eval<value_type>(first_r->Buffer(), first_x->Buffer(), nullptr, first_x.Dimensions(), strides_r, strides_x, strides_W);
  }
};

} // end of namespace grid

#endif  // GRID_TENSOR_METAL_RMS_NORM_H
