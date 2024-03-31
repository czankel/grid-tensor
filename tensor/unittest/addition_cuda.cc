//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/cuda/tensor.h>

#include <grid/tensor/binary.h>
#include <grid/tensor/cuda/binary.h>
#include <grid/tensor/base/comparison.h>  // FIXME
#include <grid/tensor/cuda/unary.h>


#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "addition.h"
#include "tensor_cuda.h"


INSTANTIATE_TYPED_TEST_SUITE_P(AdditionTestCuda, AdditionTestSuite, TensorCudaType);
