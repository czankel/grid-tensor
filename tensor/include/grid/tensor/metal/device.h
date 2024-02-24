//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_METAL_DEVICE_H
#define GRID_TENSOR_METAL_DEVICE_H

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>

#include <grid/tensor/device.h>

namespace grid::device {

class Metal : public Device
{
  Metal();

 public:
  static Metal& GetDevice();

  MTL::Buffer* NewBuffer(size_t length, MTL::ResourceOptions options)
  {
    return mtl_device_->newBuffer(length, options);
  }

 private:
  static Metal* g_device_;
  MTL::Device* mtl_device_;
};

} // end of namespace grid::device

#endif  // GRID_TENSOR_METAL_DEVICE_H



////////////////////////





#if 0
 public:
  Device();
  ~Device();

  Device(Device&) = delete;
  Device& operator=(Device&) = delete;


  MTL::Function* GetFunction();
  MTL::ComputePipelineState* GetKernel(std::string_view name)

 private:
  thread_local Device* thread_device_;
  static Device* device_;

  std::unordered_map<int32_t, MTL::CommandQueue*> queue_map_;
  std::unordered_map<int32_t, std::pair<int, MTL::CommandBuffer*>> buffer_map_;
  std::unordered_map<int32_t, MTL::ComputeCommandEncoder*> encoder_map_;
  std::unordered_map<std::string, MTL::ComputePipelineState*> kernel_map_;
  std::unordered_map<std::string, MTL::Library*> library_map_;
  std::mutex mutex_;
};


NewQueue(size_t index)
{
  const std::lock_guard<std::mutex> lock(mutex_);
  auto q = device_->newCommandQueue(MAX_BUFFERS_PER_QUEUE);

  queue_map_.insert({index, q});

}

// find library by name in the same path where the library for this function (LoadLibrary) is located.
static std::string LoadLibrary(const std::string_view name)
{
  DL_info info;
  std::string lib_name = name + ".metallib";
  int success = dladdr((void*)LoadLibrary, &info);
  if (success)
    return fs::path(info.dli_fname).remove_filename() / lib_name;

  return "";
}

{
    static MTL::Device* device;

    /*
    metal_adder* adder = new metal_adder();
    adder->init_with_device(device);
    adder->prepare_data();
    adder->send_compute_command();
    */

//    NS::Error* error;
    //this->_device = device;

    auto lib = _device->newDefaultLibrary();
    if (!lib)
    throw std::runtime_error("Failed to load default library");
}

auto GetKernel(std::string_view name)
{
  auto function = NS::String::string(name, NS::ASCIIStringEncoding);
  auto mtl_function = lib->newFunction(function);
  if(!mtl_function)
    throw std::runtime_error("Failed to load kernel");

  _addFunctionPSO = device->newComputePipelineState(mtl_function, &error);
  _CommandQueue   = _device->newCommandQueue();
}

~MetalDevice()
{
  auto pool = new_scoped_memory_pool();
  for (auto& q : queue_map_) {
    q.second->release();
  }
  for (auto& b : buffer_map_) {
    b.second.second->release();
  }
  for (auto& e : encoder_map_) {
    e.second->release();
  }
  for (auto& k : kernel_map_) {
    k.second->release();
  }
  for (auto& l : library_map_) {
    l.second->release();
  }
  device_->release();

  // per device or per allocation?
  pool_->release();}



std::shared_ptr<void> CreatePool()
{
  return std::shared_ptr<void>(NS::AutoreleasePool::alloc()->init(), [](void* ptr) {
    static_cast<NS::AutoreleasePool*>(ptr)->release();
  }
}

auto LoadLibrary(MTL::Device* device, std::string_view path)
{
  NS::Error* error;
  auto library = NS::String::string(path, NS::UTF8StringEncoding);
  auto lib = device->newLibrary(library, &error);
  return std::make_pair(lib, error);
}

#endif
