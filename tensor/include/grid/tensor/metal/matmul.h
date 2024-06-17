//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_METAL_MATMUL_H
#define GRID_TENSOR_METAL_MATMUL_H

#include "grid/util/demangle.h"

#include "device.h"
#include "kernels.h"

namespace grid {

/// MatmulOperator implements a multiplication operation for tensors
/// different ranks, such as matrix multiplication (Matmul) and vector dot-product (VecDot).
template <> class MatmulOperator<device::Metal>
{
#if 0
  MatmulOperator()
  {
    static_assert (!std::is_same_v<T, double>, "double not supported");
    if (queue_ == nullptr)
    {
      auto& dev = device::Metal::GetDevice();
      //std::string op_name(TOperator<device::Metal>::kernel_name);
      std::string op_name = grid::Demangle(typeid(T).name());
      // FIXME
      if (op_name == "int")
        op_name = "int64_t";

      kernel_ = dev.GetKernel(std::string("GeMMOperatorMM" + op_name));

      // FIXME: need to create queue per "stream" not for this class
      queue_ = dev.NewQueue();

      if (kernel_ == nullptr)
        throw std::runtime_error("Failed to find kernels");
      if (queue_ == nullptr)
        throw std::runtime_error("Failed to create queue");
    }
 private:
  // Note that dimensions are mkn: M_m_k * M_k_n -> M(m,n)
  // Note that strides for all tensors (destination and sources) are:
  //    [0] row: m -> m + 1,  [1] col: n -> n + 1
  template <typename T>
  inline void Matmul(T* dest, const T* src1, const T* src2,
                     std::span<const size_t,  3> dimensions,
                     std::span<const ssize_t, 2> strides0,
                     std::span<const ssize_t, 2> strides1,
                     std::span<const ssize_t, 2> strides2) const
  {
    if (strides0[1] <= 1 && strides1[1] <= 1 && strides2[0] <= 1)
    {
      // all tensors are contiguous
      if (strides0[0] - dimensions[1] == 0 &&
          strides1[0] - dimensions[1] == 0 &&
          strides2[1] - dimensions[1] == 0)
      {
        // FAST
        /*for (size_t i = 0; i < dimensions[0] * dimensions[2]; i++)
        {
          T sum{0};
          for (size_t k = 0; k < dimensions[1]; k++)
            sum += src1[k] * src2[k];
          dest[i] = sum;
        }
        */
      }
      // inner vector dot
      else
        throw std::runtime_error("slow gemm not supported"):
    else
      throw std::runtime_error("slow gemm not supported"):
  }

  // vecdot FIXME
  template <typename T0, typename T1, typename T2, size_t TRank>
  void operator()(T0& result, const T1& tensor1, const T2& tensor2,
                  const std::array<size_t, TRank+1>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1,
                  const std::array<ssize_t, TRank>& strides2) const
  {
    MTL::CommandBuffer* command_buffer = queue_->commandBufferWithUnretainedReferences();
    if (!command_buffer)
      throw std::runtime_error("failed to create command buffer");

    command_buffer->retain();

    MTL::ComputeCommandEncoder* enc = command_buffer->computeCommandEncoder();
    enc->retain();

    enc->setComputePipelineState(kernel_);
    enc->setBuffer(tensor1.Buffer(), 0, 0);
    enc->setBuffer(tensor2.Buffer(), 0, 1);
    enc->setBuffer(result.Buffer(), 0, 2);

    enc->setBytes(&dimensions[1], sizeof(size_t), 3);
    enc->setBytes(&result.Strides(), 2 * sizeof(size_t), 4);
    enc->setBytes(&strides1, 2 * sizeof(size_t), 5);
    enc->setBytes(&strides2, 2 * sizeof(size_t), 6);

    // FIXME: is it row/col or col/row? does it matter, i.e. align with kernel?
    MTL::Size grid_size = MTL::Size(dimensions[0], dimensions[2], 1);

  //set_array_buffer(compute_encoder, out, 1);
  //compute_encoder->setBytes(&odd, sizeof(bool), 2);


    NS::UInteger thread_group_size_ = std::min(dimensions[2], kernel_->maxTotalThreadsPerThreadgroup());
    MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);
    enc->dispatchThreads(grid_size, thread_group_size);
    enc->endEncoding();

    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    MTL::CommandBufferStatus status = command_buffer->status();
    std::cout << "status: " << status << std::endl;

    enc->release();
    command_buffer->release();
  }
    MTL::CommandBuffer* command_buffer = queue_->commandBufferWithUnretainedReferences();
    if (!command_buffer)
      throw std::runtime_error("failed to create command buffer");

    command_buffer->retain();
    //    assert(command_buffer != nullptr);
    MTL::ComputeCommandEncoder* enc = command_buffer->computeCommandEncoder();
    enc->retain();

    MTL::ComputePipelineState* kernel = kernel_;

    if constexpr (TTensor1::rank == 0 && TTensor2::rank == 0)
      kernel = kernel_ss_;
    else if constexpr (TTensor1::rank == 0)
      kernel = kernel_sv_;
    else if constexpr (TTensor2::rank == 0)
      kernel = kernel_vs_;

    enc->setComputePipelineState(kernel);
    enc->setBuffer(tensor1.Buffer(), 0, 0);
    enc->setBuffer(tensor2.Buffer(), 0, 1);
    enc->setBuffer(tensor0.Buffer(), 0, 2);

    size_t array_length = strides0[0] * dimensions[0];
    MTL::Size grid_size = MTL::Size(array_length, 1, 1);

    NS::UInteger thread_group_size_ = std::min(array_length, kernel->maxTotalThreadsPerThreadgroup());
    MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);
    enc->dispatchThreads(grid_size, thread_group_size);
    enc->endEncoding();

    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    MTL::CommandBufferStatus status = command_buffer->status();
    std::cout << "status: " << status << std::endl;

    enc->release();
    command_buffer->release();

    return;
  }
#endif


 public:
  // FIXME: output should be last?
  template<std::ranges::input_range R1,
           std::ranges::input_range R2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<R1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<O>> &&
           std::indirectly_copyable<std::ranges::iterator_t<R2>, std::ranges::iterator_t<O>>
  void operator()(O&& o, R1&& r1, R2&& r2) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first1 = std::ranges::cbegin(r1);
    auto first2 = std::ranges::cbegin(r2);
    auto result = std::ranges::begin(o);

    constexpr size_t rank1 = std::ranges::iterator_t<R1>::rank;
    constexpr size_t rank2 = std::ranges::iterator_t<R2>::rank;

    size_t dim_j = first1.Extents()[rank1 - 1];

    if (dim_j != first2.Extents()[0])
      throw std::runtime_error("GeMM: inner dimensions don't match");

    if constexpr (rank1 == 2 && rank2 == 2)
      eval<value_type>(result.Buffer(), first1.Buffer(), first2.Buffer(),
                       std::span(first1.Extents()),
                       dim_j,
                       std::span(result.Strides()),
                       std::span(first1.Strides()),
                       std::span(first2.Strides()));
    else if constexpr (rank1 == 2 && rank2 == 1)
      eval<value_type>(result.Buffer(), first1.Buffer(), first2.Buffer(),
                       std::array<size_t, 2>{first1.Extents()[0], 1},
                       dim_j,
                       std::array<const ssize_t, 2>{result.Strides()[0], 0},
                       std::span(first1.Strides()),
                       std::array<const ssize_t, 2>{first2.Strides()[0], 0}); // FIXME: wrong?
    else if constexpr (rank1 == 1 && rank2 == 2)
      eval<value_type>(result.Buffer(), first2.Buffer(), first1.Buffer(),
                       std::array<size_t, 2>{first1.Extents()[0], 1},
                       dim_j,
                       std::array<const ssize_t, 2>{result.Strides()[0], 0},
                       std::span(first2.Strides()),
                       std::array<const ssize_t, 2>{first1.Strides()[0], 0});
    else if constexpr (rank1 == 1 && rank2 == 1)
      eval<value_type>(result.Buffer(), first1.Buffer(), first2.Buffer(),
                       std::array<size_t, 2>{first1.Extents()[0], 1},
                       dim_j,
                       std::array<const ssize_t, 2>{result.Strides()[0], 0},
                       std::array<const ssize_t, 2>{first1.Strides()[0], 0},
                       std::array<const ssize_t, 2>{first2.Strides()[0], 0});
    else
      throw std::runtime("invalid matrix/vector multiplication");
  }
};

} // end of namespace grid

#endif  // GRID_TENSOR_METAL_MATMUL_H
