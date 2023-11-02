//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_DEVICE_H
#define GRID_TENSOR_DEVICE_H

/// Device describes the underlying CPU or excellerator.
/// It provides an allocator for output tensors and .... memory mapped ..
/// It is up to the implementation to optimize ..
/// FIXME: this is more of a Memory Device, which could be storage, GDS, RAM, etc.

// Allocator - allocates ...
// MemoryMappedFile / MMap --> a file as a backend for access, allocate, streaming
// CudaMemory --

namespace grid {






} // end of namespace grid



#endif  // GRID_TENSOR_DEVICE_H


