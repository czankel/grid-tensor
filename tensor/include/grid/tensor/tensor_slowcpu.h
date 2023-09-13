//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_SLOWCPU_H
#define GRID_TENSOR_TENSOR_SLOWCPU_H

#include <initializer_list>
#include <type_traits>
#include <array>
#include <algorithm>

#include "tensor.h"

namespace grid {

/// TensorSlowCpu is an unoptimized tensor implementatoin for the CPU.
template <typename, size_t, auto...> struct TensorSlowCpu;

} // end of namespace grid

#include "slowcpu/tensor.h"
#include "slowcpu/basic_operations.h"
#include "slowcpu/multiplication.h"

#endif  // GRID_TENSOR_TENSOR_SLOWCPU_H
