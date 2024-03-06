//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_METAL_UNARY_H
#define GRID_TENSOR_METAL_UNARY_H

#include <span>
#include <algorithm>
#include <ranges>

#include "../concepts.h"

namespace grid {


/// UnaryOperator<Operator> implements element-wise unary operation on a tensors.
///
///  @tparm TOperator binary operator
template <template <typename> typename TOperator>
class UnaryOperator<TOperator<device::Metal>>
{

};

} // end of namespace grid

#endif  // GRID_TENSOR_METAL_UNARY_H
