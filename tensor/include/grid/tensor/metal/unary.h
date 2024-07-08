//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_METAL_UNARY_H
#define GRID_TENSOR_METAL_UNARY_H

#include <grid/util/demangle.h>

#include "device.h"
#include "kernels.h"
#include "utils.h"

namespace grid {

// UnaryOperator<Operator> implements element-wise unary operation on a tensors for metal devices.
template <template <typename> typename TOperator>
class UnaryOperator<TOperator, device::Metal>
{
  template <typename T>
  void eval(MTL::Buffer* dst, const MTL::Buffer* src, auto dimensions, auto strides0, auto strides1) const
  {
    constexpr size_t rank = dimensions.size();

    auto& device = device::Metal::GetDevice();
    auto& encoder = device.Encoder();

    encoder->setBuffer(src, 0, 0);
    encoder->setBuffer(dst, 0, 1);

    size_t s1 = strides1.size();

    MTL::ComputePipelineState* pipeline;
    if (rank == 0 || (rank == 1 && (s1 == 0 || strides1[s1 - 1] == 1)))
    {
      std::string quantity = s1 == 0 ? "S" : "V";
      static metal::Kernel<T> kernel("UnaryOperator" + quantity + std::string(TOperator<device::Metal>::kernel_name));

      pipeline = kernel.ComputePipelineState();
      encoder->setComputePipelineState(pipeline);

      size_t array_length = 1;
      if constexpr (rank > 0)
      {
        array_length = dimensions[0];
        if (strides0.size() != 0)
          array_length *= strides0[0];
      }

      MTL::Size grid_size = MTL::Size(array_length, 1, 1);
      NS::UInteger thread_group_size_ = std::min(array_length, pipeline->maxTotalThreadsPerThreadgroup());
      MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);

      encoder.DispatchThreads(grid_size, thread_group_size);

      device.Wait(); // TODO: use callback or manage dispaltched jobs
    }
    else
    {
      static metal::Kernel<T>
        kernel("UnaryOperatorRank" + std::to_string(rank) + std::string(TOperator<device::Metal>::kernel_name));

      encoder->setBytes(strides1.data(), strides1.size() * sizeof(size_t), 3);

      auto [ grid_size, group_size] = GetBlockSize(dimensions);
      encoder.DispatchThreads(grid_size, group_size);

      device.Wait(); // TODO: use callback or manage dispaltched jobs
    }
  }


 public:
  template<std::ranges::input_range R,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<O>>
  void operator()(R&& r, O&& o) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first =  std::ranges::cbegin(r);
    auto result = std::ranges::begin(o);

    std::span strides0(result.Strides());
    std::span strides1(first.Strides());

    details::Fold([&](auto dimensions, bool contiguous) {
        if (contiguous)
          eval<value_type>(result.Buffer(), first.Buffer(), dimensions,
               strides0.template first<(dimensions.size() > 0) ? dimensions.size() - 1 : 0>(),
               strides1);
        else
          eval<value_type>(result.Buffer(), first.Buffer(), dimensions, strides0, strides1);
    }, std::span(result.Extents()), strides0, strides1);
  }
};

template <> struct CopyOperator<device::Metal> { static constexpr std::string_view kernel_name = "Copy";  };
template <> struct NegOperator<device::Metal> { static constexpr std::string_view kernel_name = "Neg";  };

} // end of namespace grid

#endif  // GRID_TENSOR_METAL_UNARY_H
