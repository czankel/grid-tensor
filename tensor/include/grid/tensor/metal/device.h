//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_METAL_DEVICE_H
#define GRID_TENSOR_METAL_DEVICE_H

#include <Metal/Metal.hpp>
#include <grid/tensor/device.h>

#include <string>

namespace grid::device {

/// Metal is a Device that implements a singleton for managing the GPU devices.
class Metal : public Device
{
  // Private constructor
  Metal();// = delete;

  Metal(Metal&) = delete;
  Metal& operator=(Metal&) = delete;

 public:
  ~Metal()
  {
    for (auto& k : kernels_) {
      k.second->release();
    }
  }

  static Metal& GetDevice();

  // TODO use smart ptr?
  MTL::Buffer* NewBuffer(size_t length, MTL::ResourceOptions options)
  {
    return mtl_device_->newBuffer(length, options);
  }

  MTL::ComputePipelineState* GetKernel(const std::string& name)
  {
    if (auto it = kernels_.find(name); it != kernels_.end())
      return it->second;

    auto function_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
    auto mtl_function = library_->newFunction(function_name);
    if (!mtl_function)
      throw std::runtime_error("Failed to find metal function: " + name);

    NS::Error* error = nullptr;
    auto kernel = mtl_device_->newComputePipelineState(mtl_function, &error);
    if (error)
      throw std::runtime_error(error->localizedDescription()->utf8String());

    kernels_.insert({name, kernel});

    return kernel;
  }

  MTL::CommandQueue* GetQueue()
  {
    if (queue_ == nullptr)
    {
      // TODO: use pool? support multiple queues (per thread/stream)? Need to lock?
      queue_ = mtl_device_->newCommandQueue();
    }
    return queue_;
  }

 private:
  static Metal* g_device_;
  MTL::Device* mtl_device_;

  MTL::Library* library_;
  std::unordered_map<std::string, MTL::ComputePipelineState*> kernels_;
  MTL::CommandQueue* queue_;
};

} // end of namespace grid::device

#endif  // GRID_TENSOR_METAL_DEVICE_H
