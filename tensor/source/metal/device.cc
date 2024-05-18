//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


#include <grid/tensor/metal/device.h>

using namespace grid::device;

Metal::Metal()
{
  auto devices = MTL::CopyAllDevices();
  mtl_device_ = static_cast<MTL::Device*>(devices->object(0)) ?: MTL::CreateSystemDefaultDevice();
  if (!mtl_device_)
    throw std::runtime_error("Failed to load device");

  auto library = NS::String::string(METAL_PATH, NS::UTF8StringEncoding);
  NS::Error* error = nullptr;
  library_ = mtl_device_->newLibrary(library, &error);
  if (error != nullptr)
    throw std::runtime_error("failed to create metal library");

#if 0 // FIXME
  NS::AutoreleasePool* pool_;
  pool_ = NS::AutoreleasePool::alloc()->init();
#endif
}


Metal& Metal::GetDevice()
{
  if (g_device_ == nullptr)
    g_device_ = new Metal();

  return *g_device_;
}

#if 0
inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}
#endif

grid::device::Metal* grid::device::Metal::g_device_;
