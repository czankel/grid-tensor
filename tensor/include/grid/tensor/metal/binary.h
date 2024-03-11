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

#include <stdio.h>
#include <grid/util/demangle.h>

#include "device.h"

namespace grid {

template <template <typename> typename TOperator, typename T>
class BinaryOperator<TOperator<device::Metal>, T>
{
 public:

  BinaryOperator()
  {
    static_assert (!std::is_same_v<T, double>, "double not supported");
    if (queue_ == nullptr)
    {
      auto& dev = device::Metal::GetDevice();
      std::string op_name(TOperator<device::Metal>::kernel_name);
      op_name += grid::Demangle(typeid(T).name());

      kernel_vv_ = dev.GetKernel(std::string("BinaryOperatorVV" + op_name));
      kernel_vs_ = dev.GetKernel(std::string("BinaryOperatorVS" + op_name));
      kernel_sv_ = dev.GetKernel(std::string("BinaryOperatorSV" + op_name));
      kernel_ss_ = dev.GetKernel(std::string("BinaryOperatorSS" + op_name));

      // FIXME: need to create queue per "stream" not for this class
      queue_ = dev.NewQueue();

      if (kernel_vv_ == nullptr || kernel_vs_ == nullptr || kernel_sv_ == nullptr || kernel_ss_ == nullptr)
        throw std::runtime_error("Failed to find kernels");
      if (queue_ == nullptr)
        throw std::runtime_error("Failed to create queue");
    }
  }

  template <typename TTensor0, typename TTensor1, typename TTensor2, size_t TRank>
  void operator()(TTensor0& result, const TTensor1& tensor1, const TTensor2& tensor2,
                  const std::array<size_t,  TRank>& dimensions,
                  const std::array<ssize_t, TRank>& strides0,
                  const std::array<ssize_t, TRank>& strides1,
                  const std::array<ssize_t, TRank>& strides2) const
  {
    //MTL::CommandBuffer* command_buffer = queue_->commandBuffer();
    MTL::CommandBuffer* command_buffer = queue_->commandBufferWithUnretainedReferences();
    if (!command_buffer)
      throw std::runtime_error("failed to create command buffer");

    command_buffer->retain();
    //    assert(command_buffer != nullptr);
    MTL::ComputeCommandEncoder* enc = command_buffer->computeCommandEncoder();
    enc->retain();

    MTL::ComputePipelineState* kernel = kernel_vv_;
    if constexpr (TTensor1::rank == 0 && TTensor2::rank == 0)
      kernel = kernel_ss_;
    else if constexpr (TTensor1::rank == 0)
      kernel = kernel_sv_;
    else if constexpr (TTensor2::rank == 0)
      kernel = kernel_vs_;

    enc->setComputePipelineState(kernel);
    enc->setBuffer(tensor1.Buffer(), 0, 0);
    enc->setBuffer(tensor2.Buffer(), 0, 1);
    enc->setBuffer(result.Buffer(), 0, 2);

    // FIXME???
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
  }

 private:
  static MTL::ComputePipelineState* kernel_vv_;
  static MTL::ComputePipelineState* kernel_vs_;
  static MTL::ComputePipelineState* kernel_sv_;
  static MTL::ComputePipelineState* kernel_ss_;
  static MTL::CommandQueue* queue_;
};

template <template <typename> typename TOperator, typename T>
MTL::ComputePipelineState* BinaryOperator<TOperator<device::Metal>, T>::kernel_vv_;
template <template <typename> typename TOperator, typename T>
MTL::ComputePipelineState* BinaryOperator<TOperator<device::Metal>, T>::kernel_vs_;
template <template <typename> typename TOperator, typename T>
MTL::ComputePipelineState* BinaryOperator<TOperator<device::Metal>, T>::kernel_sv_;
template <template <typename> typename TOperator, typename T>
MTL::ComputePipelineState* BinaryOperator<TOperator<device::Metal>, T>::kernel_ss_;
template <template <typename> typename TOperator, typename T>
MTL::CommandQueue* BinaryOperator<TOperator<device::Metal>, T>::queue_;

template <> struct AddOperator<device::Metal> { static constexpr std::string_view kernel_name = "Add";  };
template <> struct SubOperator<device::Metal> {};
template <> struct MulOperator<device::Metal> {};
template <> struct DivOperator<device::Metal> {};


} // end of namespace grid

#endif  // GRID_TENSOR_METAL_BINARY_H

