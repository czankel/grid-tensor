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
  void eval(MTL::Buffer* d, const MTL::Buffer* x, const MTL::Buffer* y,
            auto dimensions, auto strides_d, auto strides_x, auto strides_y) const
  {
    constexpr size_t rank = dimensions.size();

    auto& device = device::Metal::GetDevice();
    auto& encoder = device.Encoder();

    encoder->setBuffer(x, 0, 0);
    encoder->setBuffer(y, 0, 1);
    encoder->setBuffer(d, 0, 2);

    size_t s1 = strides_x.size();
    size_t s2 = strides_y.size();

    MTL::ComputePipelineState* pipeline;
    if (rank == 0 ||
        (rank == 1 && (s1 == 0 || strides_x[s1 - 1] == 1) && (s2 == 0 || strides_y[s2 - 1] == 1)))
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
          array_length *= strides_d[0];
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

      auto [b_strides_x, b_strides_y] = BroadcastStrides(strides_x, strides_y);
      encoder->setBytes(b_strides_x.data(), b_strides_x.size() * sizeof(size_t), 3);
      encoder->setBytes(b_strides_y.data(), b_strides_y.size() * sizeof(size_t), 4);

      auto [ grid_size, group_size] = GetBlockSize(dimensions);
      encoder.DispatchThreads(grid_size, group_size);

      device.Wait(); // TODO: use callback or manage dispaltched jobs
    }
  }

 public:
  template<std::ranges::input_range I1,
           std::ranges::input_range I2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I1>, std::ranges::iterator_t<O>> &&
           std::indirectly_copyable<std::ranges::iterator_t<I2>, std::ranges::iterator_t<O>>
  void operator()(I1&& in1, I2&& in2, O&& out) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in1);
    auto first_y = std::ranges::cbegin(in2);

    std::span strides_d(first_d.Strides());
    std::span strides_x(first_x.Strides());
    std::span strides_y(first_y.Strides());

    details::Fold([&](auto dimensions, bool contiguous) {
        if (contiguous)
          eval<value_type>(
              first_d.Buffer(), first_x.Buffer(), first_y.Buffer(), dimensions,
              strides_d.template first<(dimensions.size() > 0) ? dimensions.size() - 1 : 0>(),
              strides_x, strides_y);
        else
          eval<value_type>(first_d.Buffer(), first_x.Buffer(), first_y.Buffer(), dimensions, strides_d, strides_x, strides_y);
    }, std::span(first_d.Extents()), strides_d, strides_x, strides_y);
  }
};

template <> struct AddOperator<device::Metal> { static constexpr std::string_view kernel_name = "Add";  };
template <> struct SubOperator<device::Metal> { static constexpr std::string_view kernel_name = "Sub";  };
template <> struct MulOperator<device::Metal> { static constexpr std::string_view kernel_name = "Mul";  };
template <> struct DivOperator<device::Metal> { static constexpr std::string_view kernel_name = "Div";  };

} // end of namespace grid

#endif  // GRID_TENSOR_METAL_BINARY_H
