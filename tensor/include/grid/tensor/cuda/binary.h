//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_CUDA_BINARY_H
#define GRID_TENSOR_CUDA_BINARY_H

#include <span>
#include <algorithm>
#include <ranges>

#include "device.h"

namespace grid {

void CudaDeviceSynchronize(); // FIXME move to some header?

/// BinaryOperation<Operator> implements element-wise binary operations of two tensors.
/// The dimensions of the tensors must match following broadcasting rules.
/// The resulting rank is the maximum of the tensor ranks.
///
///  @tparm TOperator binary operator
template <template <typename> typename TOperator>
class BinaryOperation<TOperator, device::Cuda>
{
  // contiguous scalar|vector x scalar|vector operations
  template <typename T> void EvalSS(T*, const T*, const T*, size_t) const;
  template <typename T> void EvalSV(T*, const T*, const T*, size_t) const;
  template <typename T> void EvalVS(T*, const T*, const T*, size_t) const;
  template <typename T> void EvalVV(T*, const T*, const T*, size_t) const;

  // contiguous (of lower rank) with strides
  template <typename T, size_t R>
  void EvalContiguous(T*, const T*, const T*,
                      std::span<const size_t, R>,
                      std::span<const ssize_t, R>,
                      std::span<const ssize_t, R>,
                      std::span<const ssize_t, R>) const;

  // non-contiguous
  template <typename T, size_t R>
  void EvalDiscontiguous(T*, const T*, const T*,
                         std::span<const size_t, R>,
                         std::span<const ssize_t, R>,
                         std::span<const ssize_t, R>,
                         std::span<const ssize_t, R>) const;

 public:

#if !defined(__CUDACC__)

  template<std::ranges::input_range I1,
           std::ranges::input_range I2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I1>, std::ranges::iterator_t<O>> &&
           std::indirectly_copyable<std::ranges::iterator_t<I2>, std::ranges::iterator_t<O>>
  void operator()(I1&& in1, I2&& in2, O&& out) const
  {
    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in1);
    auto first_y = std::ranges::cbegin(in2);

    Fold([&](auto dimensions, auto strides_d, auto strides_x, auto strides_y) {

        if constexpr (dimensions.size() > 3)
          throw std::runtime_error("non-coontiguous tensors of rank > 3 not supported");

        constexpr size_t rank = dimensions.size();
        bool is_contiguous = IsContiguous(strides_d, strides_x, strides_y);
        size_t dim = dimensions[0];

        constexpr size_t rank_x = strides_x.size();
        constexpr size_t rank_y = strides_y.size();
        if constexpr (rank_x == 0 && rank_y == 0)
        {
          EvalSS(&*first_d, &*first_x, &*first_y, 1UL); // FIXME: what's the 1UL?
        }
        else if constexpr (rank == 1)
        {
          if (is_contiguous)
          {
            if (rank_x == 0)
              EvalSV(&*first_d, &*first_x, &*first_y, dim);
            else if (rank_y == 0)
              EvalVS(&*first_d, &*first_x, &*first_y, dim);
            else // if (rank_x != 0 && rank_y != 0)
              EvalVV(&*first_d, &*first_x, &*first_y, dim);
          }
          else
          {
            const auto [b_strides_x, b_strides_y] = BroadcastStrides(strides_x, strides_y);
            EvalDiscontiguous(&*first_d, &*first_x, &*first_y, dimensions,
                              strides_d.template first<rank>(),
                              std::span(b_strides_x).template first<rank>(),
                              std::span(b_strides_y).template first<rank>());
           }
        }
        else if constexpr (rank < first_d.Rank())
        {
          constexpr size_t r_x = std::min(strides_x.size(), rank);
          constexpr size_t r_y = std::min(strides_y.size(), rank);
          const auto [b_strides_x, b_strides_y] =
            BroadcastStrides(strides_x.template first<r_x>(), strides_y.template first<r_y>());
          EvalContiguous(&*first_d, &*first_x, &*first_y, dimensions,
                         strides_d.template first<rank>(),
                         std::span(b_strides_x), std::span(b_strides_y));
        }
        else
        {
          const auto [b_strides_x, b_strides_y] = BroadcastStrides(strides_x, strides_y);
          EvalDiscontiguous(&*first_d, &*first_x, &*first_y, dimensions,
                            strides_d, std::span(b_strides_x), std::span(b_strides_y));
        }

         CudaDeviceSynchronize(); // FIXME
    }, std::span(first_d.Extents()), std::span(first_d.Strides()),
       std::span(first_x.Strides()), std::span(first_y.Strides()));

    std::span strides_d(first_d.Strides());
    std::span strides_x(first_x.Strides());
    std::span strides_y(first_y.Strides());

  }

#endif  // !__CUDACC__
};

} // end of namespace grid

#endif // GRID_TENSOR_CUDA_BINARY_H
