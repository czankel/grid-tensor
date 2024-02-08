//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

using testing::ElementsAre;

namespace {
template <typename _T> constexpr size_t size(size_t count) { return sizeof(_T) * count; }
}


template <typename T> class SoftMaxTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(SoftMaxTestSuite);


TYPED_TEST_P(SoftMaxTestSuite, TensorSoftMaxRank1)
{
  typename TypeParam::Tensor tensor{ 1.618, 2., 3.14, 5.382, -8.5, 1.3, -2.1, 3.477, 5.5 };
  typename TypeParam::Tensor result = grid::SoftMax(tensor);
  typename TypeParam::Tensor expected
  { 0.944665670912764495e-02, 0.138413555743161758e-01, 0.432787127410648800e-01,
    0.407345162648720671e-00, 0.381141786703997885e-06, 0.687341376275671639e-02,
    0.229388293053500676e-03, 0.606221835647282256e-01, 0.458362745564445395e-00 };

  EXPECT_EQ(result, expected);
}


REGISTER_TYPED_TEST_SUITE_P(SoftMaxTestSuite,
    TensorSoftMaxRank1);
