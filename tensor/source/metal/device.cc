//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/metal/device.h>

using namespace grid::device;

Metal::Metal()
{
  auto devices = MTL::CopyAllDevices();
  mtl_device_ = static_cast<MTL::Device*>(devices->object(0)) ?: MTL::CreateSystemDefaultDevice();
  if (!mtl_device_)
    throw std::runtime_error("Failed to load device");

#if 0
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

grid::device::Metal* grid::device::Metal::g_device_;
