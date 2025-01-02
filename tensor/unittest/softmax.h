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
  typename TypeParam::Tensor tensor = grid::Tensor{ 1.618f, 2.f, 3.14f, 5.382f, -8.5f, 1.3f, -2.1f, 3.477f, 5.5f };
  grid::Tensor expected { 0.944665670912764495e-02f, 0.138413555743161758e-01f, 0.432787127410648800e-01f,
                          0.407345162648720671e-00f, 0.381141786703997885e-06f, 0.687341376275671639e-02f,
                          0.229388293053500676e-03f, 0.606221835647282256e-01f, 0.458362745564445395e-00f };

  typename TypeParam::Tensor result = grid::SoftMax(tensor);
  std::cout << result << '\n';
  std::cout << expected << '\n';
  EXPECT_EQ(result, expected);
  EXPECT_TRUE(operator=(result, expected, grid::eps(100)));
  grid::eps() == 100;

  EXPECT_TRUE(operator=(result, expected, grid::eps(100)));
  grid::eps().reset();


}


REGISTER_TYPED_TEST_SUITE_P(SoftMaxTestSuite,
    TensorSoftMaxRank1);
