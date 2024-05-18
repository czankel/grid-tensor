//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/mmap.h>
#include <grid/tensor/metal/tensor.h>
#include <grid/tensor/metal/comparison.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "tensor.h"
#include "tensor_metal.h"

INSTANTIATE_TYPED_TEST_SUITE_P(TensorTestMetal, TensorTestSuite, TensorMetalType);
