//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_CUDA_SOFTMAX_H
#define GRID_TENSOR_CUDA_SOFTMAX_H

#include <span>
#include <algorithm>
#include <ranges>

#include "../precision.h"
#if 0
namespace {
template <typename t>
void printer(t* data)
{
  for (int i = 0; i < 10; i++)
    printf("%f, ", (float)data[i]);
  printf("\n");
}
}
#endif

namespace grid {

void CudaDeviceSynchronize(); // FIXME move to some header?

// FIXME: this is just boilerplate code??

/// SoftMaxOperator implements the softmax operation
///
///  @tparm TOperator binary operator
template <> class SoftMaxOperator<device::Cuda>
{
  template <typename T, size_t R>
  void EvalContiguous(T*, const T*,
                      std::span<const size_t, R>,
                      std::span<const ssize_t, R>,
                      std::span<const ssize_t, R>) const;

  template <typename T, size_t R>
  void EvalDiscontiguous(T*, const T*,
                         std::span<const size_t, R>,
                         std::span<const ssize_t, R>,
                         std::span<const ssize_t, R>) const;

 public:

#if !defined(__CUDACC__)

  template<std::ranges::input_range I,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);
printf("rankd %lu %lu\n", in.rank, out.rank);
    std::span strides_d(first_d.Strides());
    std::span strides_x(first_x.Strides());
        printf("dim %lu %lu\n", first_d.Extents()[0], first_d.Extents()[1]);
        printf("stridesd %lu %lu %lu\n", strides_d[0], strides_d[1], strides_d.size());
        printf("stridesx %lu %lu %lu\n", strides_x[0], strides_x[1], strides_x.size());

    Fold([&](auto folded_dims, const bool contiguous) {

        constexpr size_t folded_rank = folded_dims.size();
        constexpr size_t rank = std::ranges::iterator_t<I>::rank;

        if constexpr (folded_rank > 3)
          throw std::runtime_error("non-coontiguous tensors of rank > 3 not supported");

        if (contiguous)
          EvalContiguous(&*first_d, &*first_x,
                         folded_dims,
                         strides_d.template first<folded_rank>(),
                         strides_x.template first<folded_rank>());
        else if constexpr (rank > 0)  // rank == 0 is always contiguous
          EvalDiscontiguous(&*first_d, &*first_x, folded_dims, strides_d, strides_x);

        CudaDeviceSynchronize(); // FIXME
    printf("SoftMax Cuda\n");
    printer(out.Data());
    }, std::span(first_d.Extents()), strides_d, strides_x);
  }

#endif  // !__CUDACC__
};

} // end of namespace grid

#endif // GRID_TENSOR_CUDA_SOFTMAX_H
