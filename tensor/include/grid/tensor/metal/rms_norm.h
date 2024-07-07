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

// FIXME
#if 1
static constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static constexpr int REDUCE_N_READS = 16;
static constexpr int SOFTMAX_N_READS = 4;
static constexpr int RMS_N_READS = 4;
static constexpr int RMS_LOOPED_LIMIT = 4096;
#endif

template <> class RmsNormOperator<device::Metal>
{

  template <typename T>
  void eval(MTL::Buffer* y, const MTL::Buffer* x, auto dimensions, auto strides0, auto strides1) const
  {

  }


  template <typename T>
  void eval(MTL::Buffer* y, const MTL::Buffer* x, const MTL::Buffer* W,
            auto dimensions, auto strides0, auto strides1, auto strides2) const
  {
    // FIXME constexpr size_t rank = dimensions.size();

    // FIXME
    const int simd_size = 32;   // FIXME
    const int line_limit = 4096;  // FIXME is this thread-count? size of

    auto& device = device::Metal::GetDevice();
    auto& encoder = device.Encoder();

    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(W, 0, 1);
    encoder->setBuffer(y, 0, 2);

    size_t row_size = dimensions.back();
    int n_rows = std::accumulate(std::begin(dimensions), std::end(dimensions) - 1, std::multiplies<int>());

    MTL::ComputePipelineState* pipeline;
    size_t threadgroup_size;

    if (row_size > line_limit)
    {
      static metal::Kernel<T> kernel("RmsNormLoop");
      pipeline = kernel.ComputePipelineState();
      threadgroup_size = (((row_size + RMS_N_READS - 1) & -RMS_N_READS) + simd_size) & -simd_size;
    }
    else
    {
      static metal::Kernel<T> kernel("RmsNormLine");
      pipeline = kernel.ComputePipelineState();
      threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
    }

    MTL::Size grid_size = MTL::Size(n_rows * threadgroup_size, 1, 1);
    MTL::Size group_size = MTL::Size(threadgroup_size, 1, 1);

    uint32_t w_stride = strides2.back();

    // FIXME encoder.SetInputArray(W, 1); setBuffer above?
    // FIXME encoder.SetOutputArray(out, 2);; setBuffer above?
    encoder->setComputePipelineState(pipeline);
    // FIXME encoder->setBytes(&eps, ...);
    // FIXME encoder->setBytes(&dimensions[0], sizeof(int), 4);
    encoder->setBytes(&w_stride, sizeof(uint32_t), 5);
    // FIXME encoder->setThreadgroupMemoryLength(...16 * 8, 0); // FIXME
    encoder->setThreadgroupMemoryLength(simd_size * sizeof(float), 1); // FIXME float
    encoder.DispatchThreads(grid_size, group_size);

    device.Wait();  // FIXME use callback / manage streams
  }

 public:

  // y = RMS(x)
  template<std::ranges::input_range R, std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<O>>
  void operator()(R&& r, O&& o) const
  {
#if 0
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first = std::ranges::cbegin(r);
    auto result = std::ranges::begin(o);
#endif
    // FIXME: add strides, etc. eval<value_type>(result.Buffer(), first.Buffer());
  }


  // y = RMS(x) * W
  template<std::ranges::input_range R1, std::ranges::input_range R2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<O>>
  void operator()(R1&& r1, R2&& r2, O&& o) const
  {
    auto first1 = std::ranges::cbegin(r1);
    auto first2 = std::ranges::cbegin(r2);
    auto result = std::ranges::begin(o);

    std::span strides0(result.Strides());
    std::span strides1(first1.Strides());
    std::span strides2(first2.Strides());

    //eval(MTL::Buffer* y, const MTL::Buffer* x, const MTL::Buffer* W,
     //       auto dimensions, auto strides0, auto strides1, auto strides2) const
    //eval();
#if 0
    using tensor_type = std::remove_cvref_t<O>;
    using value_type = tensor_type::value_type;
    constexpr value_type eps = Eps<value_type>::default_value;

    auto first = std::ranges::cbegin(r);
    auto result = std::ranges::begin(o);
    auto& extents = result.Extents();

    auto [value, count] = SumSquare(&*first, std::span(extents), std::span(first.Strides()));

    value_type scale = 1.0f/sqrtf(value / count + eps);
    BinaryOperator<MulOperator<device::Metal>>()(r, Tensor(scale), o);
#endif
  }
};

} // end of namespace grid

#endif  // GRID_TENSOR_METAL_RMS_NORM_H
