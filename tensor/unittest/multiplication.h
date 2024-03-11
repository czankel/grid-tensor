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
  typename TypeParam::Tensor tensor1 = grid::Tensor({ 2,   3,   5 });
  typename TypeParam::Tensor tensor2 = grid::Tensor({ 7,  11,  13 });

  auto op = grid::Matmul(tensor1, tensor2);
  auto res = op();
  EXPECT_EQ(res, grid::Tensor{112});
}


TYPED_TEST_P(MultiplicationTestSuite, TensorMatmul)
{
  typename TypeParam::Tensor tensor1 =
    grid::Tensor{ { 3.f, 6.f, 9.f },
                  { 2.f, 8.f, 4.f },
                  { 5.f, 1.f, 7.f } };
  typename TypeParam::Tensor tensor2 =
    grid::Tensor{ { 1.f, 8.f },
                  { 5.f, 3.f },
                  { 7.f, 4.f } };
  auto expected =
    grid::Tensor{ { 3.f * 1.f + 6.f * 5.f + 9.f * 7.f, 3.f * 8.f + 6.f * 3.f + 9.f * 4.f },  // 96, 78
                  { 2.f * 1.f + 8.f * 5.f + 4.f * 7.f, 2.f * 8.f + 8.f * 3.f + 4.f * 4.f },  // 70, 56
                  { 5.f * 1.f + 1.f * 5.f + 7.f * 7.f, 5.f * 8.f + 1.f * 3.f + 7.f * 4.f } };// 59, 71

  auto op = grid::Matmul(tensor1, tensor2);
  auto res = op();
  std::cout << res << std::endl;
  std::cout << expected << std::endl;
  EXPECT_EQ(res, expected);

}

#if 0
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
#endif

REGISTER_TYPED_TEST_SUITE_P(MultiplicationTestSuite,
    TensorVecDot,
    TensorMatmul);
#if 0
    TensorScaleRight,
    TensorScalexLeft,
    TensorElemMulRank1,
    TensorElemMulRank2,
    TensorElemMulRank2Broadcast);
#endif
