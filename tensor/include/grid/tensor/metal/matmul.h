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

#include "../device.h"

namespace grid {


/// MatmulOperator implements a multiplication operation for tensors
/// different ranks, such as matrix multiplication (Matmul) and vector dot-product (VecDot).
template <typename T> class MatmulOperator<device::Metal, T>
{
 public:

  MatmulOperator()
  {
    static_assert (!std::is_same_v<T, double>, "double not supported");
    if (queue_ == nullptr)
    {
      auto& dev = device::Metal::GetDevice();
      //std::string op_name(TOperator<device::Metal>::kernel_name);
      std::string op_name = grid::Demangle(typeid(T).name());

      kernel_ = dev.GetKernel(std::string("GemmOperatorMM" + op_name));

      // FIXME: need to create queue per "stream" not for this class
      queue_ = dev.NewQueue();

      if (kernel_ == nullptr)
        throw std::runtime_error("Failed to find kernels");
      if (queue_ == nullptr)
        throw std::runtime_error("Failed to create queue");
    }
  }
#if 0
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
#endif
 public:
  // vecdot
  template <typename T0, typename T1, typename T2>
  void operator()(T0& tensor0, const T1& tensor1, const T2& tensor2,
                  const size_t dimensions, const ssize_t strides1, const ssize_t strides2) const
  {
    T* dst = tensor0.Data();
    const T* src1 = tensor1.Data();
    const T* src2 = tensor2.Data();
    T sum{0};
    if (strides1 == 1 && strides2 == 1)
    {
      for (size_t i = 0; i < dimensions; i++)
        sum += src1[i] * src2[i];
    }
    else
    {
      for (size_t i = 0; i < dimensions; i++)
      {
        sum += *src1 * *src2;
        src1 += strides1;
        src2 += strides2;
      }
    }
    *dst = sum;
  }

  // matmul
  template <typename T0, typename T1, typename T2, size_t TRank>
  requires (TRank > 1)
  void operator()(T0& tensor0, const T1& tensor1, const T2& tensor2,
                  const std::array<size_t, TRank+1>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1,
                  const std::array<ssize_t, TRank>& strides2) const
  {
    MTL::CommandBuffer* command_buffer = queue_->commandBufferWithUnretainedReferences();
    if (!command_buffer)
      throw std::runtime_error("failed to create command buffer");

    command_buffer->retain();
    //    assert(command_buffer != nullptr);
    MTL::ComputeCommandEncoder* enc = command_buffer->computeCommandEncoder();
    enc->retain();

    MTL::ComputePipelineState* kernel = kernel_;
#if 0
    if constexpr (TTensor1::rank == 0 && TTensor2::rank == 0)
      kernel = kernel_ss_;
    else if constexpr (TTensor1::rank == 0)
      kernel = kernel_sv_;
    else if constexpr (TTensor2::rank == 0)
      kernel = kernel_vs_;
#endif

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

  static MTL::ComputePipelineState* kernel_;
  static MTL::CommandQueue* queue_;
};


} // end of namespace grid

#endif  // GRID_TENSOR_METAL_MATMUL_H
