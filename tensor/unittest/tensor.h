//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

using testing::ElementsAre;

using grid::view::Slice;
using grid::view::Null;
using grid::view::NewAxis;


// Use Google's Type-Parameterized Tests so these tests can be re-used for other device implementations.

template <typename T> class TensorTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(TensorTestSuite);

#if 0
// FIXME: Scalar...
TYPED_TEST_P(TensorTestSuite, TensorBraceInitializationRank0Integer)
{
  typename TypeParam::Tensor tensor{ 4 };
  EXPECT_EQ(tensor.Rank(), 0);
}
#endif

TYPED_TEST_P(TensorTestSuite, TensorAllocInitializedRank1Double)
{
  typename TypeParam::Tensor tensor1(4UL, 1.2);

  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(1));

  double verify[] = { 1.2, 1.2, 1.2, 1.2 };
  EXPECT_EQ(memcmp(tensor1.Data(), verify, sizeof(verify)), 0);
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedRank1Double)
{
  typename TypeParam::Tensor tensor1(5UL, grid::Uninitialized<double>{});
  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(5));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(1));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocInitializedRank2Char)
{
  typename TypeParam::Tensor tensor1(5UL, 4UL, (char)'3');

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(5, 4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(4, 1));

  char verify[] = { '3', '3', '3', '3', '3',
                    '3', '3', '3', '3', '3',
                    '3', '3', '3', '3', '3',
                    '3', '3', '3', '3', '3' };
  EXPECT_EQ(memcmp(tensor1.Data(), verify, sizeof(verify)), 0);
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedRank2Double)
{
  typename TypeParam::Tensor tensor1(7UL, 3UL, grid::Uninitialized<int>{});

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(7, 3));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(3, 1));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocInitializedRank3Double)
{
  typename TypeParam::Tensor tensor1{{4, 5, 7}, 3.3};

  EXPECT_EQ(tensor1.Rank(), 3);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(4, 5, 7));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(7 * 5, 7, 1));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedRank3Double)
{
  typename TypeParam::Tensor t33({3, 2, 1}, grid::Uninitialized<double>{});
  EXPECT_THAT(t33.Strides(), ElementsAre(2 * 1, 1, 0));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedPattedRank3Double)
{
  typename TypeParam::Tensor tensor1({3, 2, 1}, {2 * 2 * 4, 2 * 2, 2}, grid::Uninitialized<double>{});
  EXPECT_EQ(tensor1.Rank(), 3);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(3, 2, 1));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(2 * 2 * 4, 2 * 2, 2));
}


TYPED_TEST_P(TensorTestSuite, TensorBroadcast)
{
  typename TypeParam::Tensor tensor(4UL, 5UL, 1.1);

  // tensor[newaxis] -> (1, 5, 4)
  auto view_newaxis_0 = tensor.View(NewAxis);
  EXPECT_EQ(view_newaxis_0.Rank(), 3);
  EXPECT_THAT(view_newaxis_0.Dimensions(), ElementsAre(1UL, 4UL, 5UL));
  EXPECT_THAT(view_newaxis_0.Strides(), ElementsAre(0, 5, 1));

  // tensor[:,newaxis] -> (4, 1, 5)
  auto view_newaxis_1 = tensor.View(Slice(), NewAxis);
  EXPECT_EQ(view_newaxis_1.Rank(), 3);
  EXPECT_THAT(view_newaxis_1.Dimensions(), ElementsAre(4UL, 1UL, 5UL));
  EXPECT_THAT(view_newaxis_1.Strides(), ElementsAre(5, 0, 1));

  // tensor[:,:,newaxis] -> (4, 5, 1)
  auto view_newaxis_2 = tensor.View(Slice(), Slice(), NewAxis);
  EXPECT_EQ(view_newaxis_2.Rank(), 3);
  EXPECT_THAT(view_newaxis_2.Dimensions(), ElementsAre(4UL, 5UL, 1UL));
  EXPECT_THAT(view_newaxis_2.Strides(), ElementsAre(5, 1, 0));

  // tensor[:,:1:0] -> (4, 1)
  auto view_change_to_broadcast = tensor.View(Slice(), Slice(0, 1, 0));
  EXPECT_EQ(view_change_to_broadcast.Rank(), 2);
  EXPECT_THAT(view_change_to_broadcast.Dimensions(), ElementsAre(4UL, 1UL));
  EXPECT_THAT(view_change_to_broadcast.Strides(), ElementsAre(5, 0));
}


REGISTER_TYPED_TEST_SUITE_P(TensorTestSuite,
    TensorAllocInitializedRank1Double,
    TensorAllocUninitializedRank1Double,
    TensorAllocInitializedRank2Char,
    TensorAllocUninitializedRank2Double,
    TensorAllocInitializedRank3Double,
    TensorAllocUninitializedRank3Double,
    TensorAllocUninitializedPattedRank3Double,
    TensorBroadcast);
