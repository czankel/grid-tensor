//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/metal/tensor.h>

#include <grid/tensor/binary_function.h>
#include <grid/tensor/metal/binary_ops.h>
// FIXME ... metal
#include <grid/tensor/base/comparison.h>


#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "addition.h"
#include "tensor_metal.h"


INSTANTIATE_TYPED_TEST_SUITE_P(AdditionTestMetal, AdditionTestSuite, TensorMetalType);
