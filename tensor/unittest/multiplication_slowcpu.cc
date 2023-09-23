//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/slowcpu/tensor.h>
#include <grid/tensor/slowcpu/comparison.h>
#include <grid/tensor/slowcpu/multiplication.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "multiplication.h"
#include "tensor_slowcpu.h"

INSTANTIATE_TYPED_TEST_SUITE_P(MultiplicationTestSlowCpu, MultiplicationTestSuite, TensorSlowCpuSuite);
