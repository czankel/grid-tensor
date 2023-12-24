//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

using testing::ElementsAre;

// helper to get the size of a type of (optional) array of COUNT elements or the TYPE:
//  size_t bytes = size<TYPE>(COUNT)
namespace {
template <typename T> constexpr size_t size(size_t count) { return sizeof(T) * count; }
}

template <typename T> class AdditionTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(AdditionTestSuite);


TYPED_TEST_P(AdditionTestSuite, TensorAddRank0)
{
  typename TypeParam::Tensor tensor1{ 5 };
  typename TypeParam::Tensor tensor2{ 3 };
  auto op = grid::TensorAdd(tensor1, tensor2);
  auto res = op();
  EXPECT_EQ(res.Rank(), 0);
  typename TypeParam::Tensor expected{ 8 };
  EXPECT_EQ(res, typename TypeParam::Tensor{ 8 });
}

TYPED_TEST_P(AdditionTestSuite, TensorAdd)
{
  typename TypeParam::Tensor t11{ 11, 22, 33, 44, 55, 66 };
  typename TypeParam::Tensor t12{ 89, 78, 67, 56, 45, 34 };
  int v1[] = { 100, 100, 100, 100, 100, 100 };

  auto op1a = grid::TensorAdd(t11, t12);
  auto res1a = op1a();
  EXPECT_EQ(memcmp(res1a.Data(), v1, sizeof(v1)), 0);

  auto&& op1b = t12 + t11;
  auto res1b = op1b();
  EXPECT_EQ(memcmp(res1b.Data(), v1, sizeof(v1)), 0);

  typename TypeParam::Tensor t21(4UL, 5UL, 1.1);
  typename TypeParam::Tensor t22(4UL, 5UL, 2.4);

  auto op2 = std::move(t21) + std::move(t22);
  auto res2 = op2();
  typename TypeParam::Tensor v2{
    { 3.5, 3.5, 3.5, 3.5, 3.5 },
    { 3.5, 3.5, 3.5, 3.5, 3.5 },
    { 3.5, 3.5, 3.5, 3.5, 3.5 },
    { 3.5, 3.5, 3.5, 3.5, 3.5 }
  };
  EXPECT_EQ(res2, v2);
}

TYPED_TEST_P(AdditionTestSuite, TensorAddAdd)
{
  typename TypeParam::Tensor tensor1(4UL, 3UL, 2.1);
  typename TypeParam::Tensor tensor2(4UL, 3UL, 1.3);
  typename TypeParam::Tensor tensor3(4UL, 3UL, 2.2);
  typename TypeParam::Tensor expected{ { 5.6, 5.6, 5.6 },
                                       { 5.6, 5.6, 5.6 },
                                       { 5.6, 5.6, 5.6 },
                                       { 5.6, 5.6, 5.6 }};
  auto&& oper = tensor1 + tensor2 + tensor3;
  auto result = oper();
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(AdditionTestSuite, TensorAddMatVecBroadcast)
{
  typename TypeParam::Tensor tensor1(4UL, 1.1);
  typename TypeParam::Tensor tensor2(4UL, 5UL, 4.4);
  typename TypeParam::Tensor result = tensor2 + tensor1.View(grid::view::Slice{}, grid::view::NewAxis);
  typename TypeParam::Tensor expected(4UL, 5UL, 5.5);
  EXPECT_EQ(result, expected);
}


REGISTER_TYPED_TEST_SUITE_P(AdditionTestSuite,
    TensorAddRank0,
    TensorAdd,
    TensorAddAdd,
    TensorAddMatVecBroadcast);
