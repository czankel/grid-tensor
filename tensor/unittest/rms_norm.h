//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

using testing::ElementsAre;

namespace {
template <typename T> constexpr size_t size(size_t count) { return sizeof(T) * count; }
}

template <typename T> class RmsNormTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(RmsNormTestSuite);


TYPED_TEST_P(RmsNormTestSuite, TensorNormRmsRank1)
{
  typename TypeParam::Tensor tensor = grid::Tensor{ 1.618f, 2.f, 3.14f, 5.382f, -8.5f, 13.f, -21.f, 34.77f, 55.f };
  float scale = 23.47965324677914722429f;
  grid::Tensor expected{ 1.618f / scale,    2.f / scale, 3.14f / scale,
                         5.382f / scale,  -8.5f / scale,  13.f / scale,
                          -21.f / scale, 34.77f / scale,  55.f / scale };

  typename TypeParam::Tensor result = grid::RmsNorm(tensor);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(RmsNormTestSuite, TensorNormRmsRank2)
{
  typename TypeParam::Tensor tensor = grid::Tensor{ { 1.618f, 2.f, 3.14f, 5.382f, -8.5f},
                                                    { 13.f, -21.f, 34.77f, 55.f, 43.5f} };
  typename TypeParam::Tensor result = grid::RmsNorm(tensor);
  // TODO: check actual result
  EXPECT_EQ(result.Rank(), 2);
}

REGISTER_TYPED_TEST_SUITE_P(RmsNormTestSuite,
    TensorNormRmsRank1,
    TensorNormRmsRank2);
