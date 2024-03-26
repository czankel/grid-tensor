//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/base/tensor.h>
#include <grid/tensor/base/binary.h>
#include <grid/tensor/base/comparison.h>
#include <grid/tensor/base/matmul.h>
#include <grid/tensor/base/unary.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "multiplication.h"
#include "tensor_base.h"

INSTANTIATE_TYPED_TEST_SUITE_P(MultiplicationTestBase, MultiplicationTestSuite, TensorBaseType);
