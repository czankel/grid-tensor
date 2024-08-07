//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

using testing::ElementsAre;

template <typename T> class UnaryTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(UnaryTestSuite);


TYPED_TEST_P(UnaryTestSuite, TensorUnaryElementaryRank0)
{
  typename TypeParam::Tensor tensor = grid::Tensor{ 5 };

  typename TypeParam::Tensor copy = Copy(tensor);
  EXPECT_EQ(copy, grid::Tensor{ 5 });

  typename TypeParam::Tensor neg = Neg(tensor);
  EXPECT_EQ(neg, grid::Tensor{ -5 });
}

TYPED_TEST_P(UnaryTestSuite, TensorUnaryElementaryRank1)
{
  typename TypeParam::Tensor tensor = grid::Tensor{ 11, 22, 33, 44, 55, 66 };

  typename TypeParam::Tensor copy = Copy(tensor);
  EXPECT_EQ(copy, (grid::Tensor{ 11, 22, 33, 44, 55, 66 }));

  typename TypeParam::Tensor neg = Neg(tensor);
  EXPECT_EQ(neg, (grid::Tensor{ -11, -22, -33, -44, -55, -66 }));
}

TYPED_TEST_P(UnaryTestSuite, TensorUnaryElementaryRank2)
{
  typename TypeParam::Tensor tensor = grid::Tensor{ {11, 22}, {33, 44}, {55, 66} };

  typename TypeParam::Tensor copy = Copy(tensor);
  EXPECT_EQ(copy, (grid::Tensor{ {11, 22}, {33, 44}, {55, 66} }));

  typename TypeParam::Tensor neg = Neg(tensor);
  EXPECT_EQ(neg, (grid::Tensor{ {-11, -22}, {-33, -44}, {-55, -66} }));
}

TYPED_TEST_P(UnaryTestSuite, TensorUnaryElementaryRank3)
{
  typename TypeParam::Tensor tensor({400, 300, 500}, 2.1f);

  typename TypeParam::Tensor copy = Copy(tensor);
  EXPECT_EQ(copy, (grid::Tensor({400, 300, 500}, 2.1f)));

  typename TypeParam::Tensor neg = Neg(tensor);
  EXPECT_EQ(neg, (grid::Tensor({400, 300, 500}, -2.1f)));
}

REGISTER_TYPED_TEST_SUITE_P(UnaryTestSuite,
    TensorUnaryElementaryRank0,
    TensorUnaryElementaryRank1,
    TensorUnaryElementaryRank2,
    TensorUnaryElementaryRank3);
