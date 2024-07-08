//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_METAL_BINARY_H
#define GRID_TENSOR_METAL_BINARY_H

#include <grid/util/demangle.h>

#include "device.h"
#include "kernels.h"
#include "utils.h"

namespace grid {

template <template <typename> typename TOperator>
class BinaryOperator<TOperator, device::Metal>
{
  template <typename T>
  void eval(MTL::Buffer* dest, const MTL::Buffer* src1, const MTL::Buffer* src2,
            auto dimensions, auto strides0, auto strides1, auto strides2) const
  {
    constexpr size_t rank = dimensions.size();

    auto& device = device::Metal::GetDevice();
    auto& encoder = device.Encoder();

    encoder->setBuffer(src1, 0, 0);
    encoder->setBuffer(src2, 0, 1);
    encoder->setBuffer(dest, 0, 2);

    size_t s1 = strides1.size();
    size_t s2 = strides2.size();

    MTL::ComputePipelineState* pipeline;
    if (rank == 0 ||
        (rank == 1 && (s1 == 0 || strides1[s1 - 1] == 1) && (s2 == 0 || strides2[s2 - 1] == 1)))
    {
      // TODO find a way to use constexpr with strings
      std::string quantities = s1 == 0 && s2 == 0? "SS" : s1 == 0? "SV" : s2 == 0? "VS" : "VV";
      static metal::Kernel<T>
        kernel("BinaryOperator" + quantities + std::string(TOperator<device::Metal>::kernel_name));

      pipeline = kernel.ComputePipelineState();
      encoder->setComputePipelineState(pipeline);

      size_t array_length = 1;
      if constexpr (rank > 0)
      {
        array_length = dimensions[0];
        if (s1 != 0)
          array_length *= strides0[0];
      }

      MTL::Size grid_size = MTL::Size(array_length, 1, 1);
      NS::UInteger threads_per_group = std::min(array_length, pipeline->maxTotalThreadsPerThreadgroup());
      MTL::Size thread_group_size = MTL::Size(threads_per_group, 1, 1);

      encoder.DispatchThreads(grid_size, thread_group_size);

      device.Wait(); // TODO: use callback or manage dispaltched jobs
    }
    else
    {
      static metal::Kernel<T>
        kernel("BinaryOperatorRank" + std::to_string(rank) + std::string(TOperator<device::Metal>::kernel_name));

      pipeline = kernel.ComputePipelineState();
      encoder->setComputePipelineState(pipeline);

      auto [b_strides1, b_strides2] = BroadcastStrides(strides1, strides2);
      encoder->setBytes(b_strides1.data(), b_strides1.size() * sizeof(size_t), 3);
      encoder->setBytes(b_strides2.data(), b_strides2.size() * sizeof(size_t), 4);

      auto [ grid_size, group_size] = GetBlockSize(dimensions);
      encoder.DispatchThreads(grid_size, group_size);

      device.Wait(); // TODO: use callback or manage dispaltched jobs
    }
  }

 public:
  template<std::ranges::input_range R1,
           std::ranges::input_range R2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<O>> &&
           std::indirectly_copyable<std::ranges::iterator_t<R2>, std::ranges::iterator_t<O>>
  void operator()(R1&& r1, R2&& r2, O&& o) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first1 = std::ranges::cbegin(r1);
    auto first2 = std::ranges::cbegin(r2);
    auto result = std::ranges::begin(o);

    std::span strides0(result.Strides());
    std::span strides1(first1.Strides());
    std::span strides2(first2.Strides());

    details::Fold([&](auto dimensions, bool contiguous) {
        if (contiguous)
          eval<value_type>(
              result.Buffer(), first1.Buffer(), first2.Buffer(), dimensions,
              strides0.template first<(dimensions.size() > 0) ? dimensions.size() - 1 : 0>(),
              strides1, strides2);
        else
          eval<value_type>(result.Buffer(), first1.Buffer(), first2.Buffer(), dimensions, strides0, strides1, strides2);
    }, std::span(result.Extents()), strides0, strides1, strides2);
  }
};

template <> struct AddOperator<device::Metal> { static constexpr std::string_view kernel_name = "Add";  };
template <> struct SubOperator<device::Metal> { static constexpr std::string_view kernel_name = "Sub";  };
template <> struct MulOperator<device::Metal> { static constexpr std::string_view kernel_name = "Mul";  };
template <> struct DivOperator<device::Metal> { static constexpr std::string_view kernel_name = "Div";  };

} // end of namespace grid

#endif  // GRID_TENSOR_METAL_BINARY_H
