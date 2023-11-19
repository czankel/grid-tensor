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

template <typename T> class RmsNormTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(RmsNormTestSuite);


TYPED_TEST_P(RmsNormTestSuite, TensorNormRmsRank1)
{
  typename TypeParam::Tensor tensor{ 1.618, 2., 3.14, 5.382, -8.5, 13., -21., 34.77, 55. };
  typename TypeParam::Tensor result = grid::TensorRmsNorm(tensor);

  double scale = 23.47965240478515269729;
  typename TypeParam::Tensor expected{ 1.618 / scale, 2. / scale, 3.14 / scale,
                                       5.382 / scale, -8.5 / scale, 13. / scale,
                                       -21. / scale, 34.77 / scale, 55. / scale };
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(RmsNormTestSuite, TensorNormRmsRank2)
{
  typename TypeParam::Tensor tensor{ { 1.618, 2., 3.14, 5.382, -8.5}, { 13., -21., 34.77, 55., 43.5} };
  auto op = grid::TensorRmsNorm(tensor);
  auto result = op();

  // TODO: check actual result
  EXPECT_EQ(result.Rank(), 2);
}

REGISTER_TYPED_TEST_SUITE_P(RmsNormTestSuite,
    TensorNormRmsRank1,
    TensorNormRmsRank2);
