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

template <typename T> class MultiplicationTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(MultiplicationTestSuite);


TYPED_TEST_P(MultiplicationTestSuite, TensorVecDot)
{
  // dot -> 14+33+65 = 112
  typename TypeParam::Tensor tensor1{   2,   3,   5 };
  typename TypeParam::Tensor tensor2{   7,  11,  13 };

  auto op = grid::MatMul(tensor1, tensor2);
  auto res = op();
  EXPECT_EQ(res, typename TypeParam::Tensor{112});
}


TYPED_TEST_P(MultiplicationTestSuite, TensorMatMul)
{
  typename TypeParam::Tensor tensor1{ { 3, 6, 9 },
                                      { 2, 8, 4 },
                                      { 5, 1, 7 }};
  typename TypeParam::Tensor tensor2{ { 1, 8 },
                                      { 5, 3 },
                                      { 7, 4 } };
  typename TypeParam::Tensor expected{{ 3 * 1 + 6 * 5 + 9 * 7, 3 * 8 + 6 * 3 + 9 * 4 }, // 96, 78
                                      { 2 * 1 + 8 * 5 + 4 * 7, 2 * 8 + 8 * 3 + 4 * 4 }, // 70, 56
                                      { 5 * 1 + 1 * 5 + 7 * 7, 5 * 8 + 1 * 3 + 7 * 4 }};// 59, 71

  auto op = grid::MatMul(tensor1, tensor2);
  auto res = op();
  EXPECT_EQ(res, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorScaleRight)
{
  typename TypeParam::Tensor tensor1{ { 4.4, 6.6, 8.8 }, { 7.7, 5.5, 3.3 } };
  typename TypeParam::Tensor tensor2{ 1.0/11. };

  typename TypeParam::Tensor result = tensor1 * tensor2;
  typename TypeParam::Tensor expected{ { 0.4, 0.6, 0.8 }, { 0.7, 0.5, 0.3 } };
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorScalexLeft)
{
  typename TypeParam::Tensor tensor1{ 1.0/11. };
  typename TypeParam::Tensor tensor2{ { 4.4, 6.6, 8.8 }, { 7.7, 5.5, 3.3 } };

  typename TypeParam::Tensor result = tensor1 * tensor2;
  typename TypeParam::Tensor expected{ { 0.4, 0.6, 0.8 }, { 0.7, 0.5, 0.3 } };
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorElemMulRank1)
{
  typename TypeParam::Tensor tensor1 { 3, 6, 9, 2, 8, 4 };
  typename TypeParam::Tensor tensor2 { 1, 8, 5, 3, 7, 4 };
  typename TypeParam::Tensor expected{ 3,48,45, 6,56,16 };

  typename TypeParam::Tensor result = grid::Mul(tensor1, tensor2);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorElemMulRank2)
{
  typename TypeParam::Tensor tensor1 { { 3, 6, 9 }, { 2, 8, 4 }};
  typename TypeParam::Tensor tensor2 { { 1, 8, 5 }, { 3, 7, 4 }};
  typename TypeParam::Tensor expected{ { 3,48,45 }, { 6,56,16 }};

  typename TypeParam::Tensor result = grid::Mul(tensor1, tensor2);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorElemMulRank2Broadcast)
{
  typename TypeParam::Tensor tensor1 { { 3, 6, 9 }, { 2, 8, 4 }};
  typename TypeParam::Tensor tensor2 { { 1, 8, 5 } };
  typename TypeParam::Tensor expected{ { 3,48,45 }, { 2,64,20 }};

  typename TypeParam::Tensor result = grid::Mul(tensor1, tensor2);
  EXPECT_EQ(result, expected);
}


REGISTER_TYPED_TEST_SUITE_P(MultiplicationTestSuite,
    TensorVecDot,
    TensorMatMul,
    TensorScaleRight,
    TensorScalexLeft,
    TensorElemMulRank1,
    TensorElemMulRank2,
    TensorElemMulRank2Broadcast);
