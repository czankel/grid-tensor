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
template <typename _T> constexpr size_t size(size_t count = 1) { return sizeof(_T) * count; }
}

// Use Google's Type-Parameterized Tests so these tests can be re-used for other device implementations.

template <typename T> class TensorTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(TensorTestSuite);


TYPED_TEST_P(TensorTestSuite, TensorBraceInitializationRank0Integer)
{
  typename TypeParam::Tensor tensor{ 4 };
  EXPECT_EQ(tensor.Rank(), 0);
}

TYPED_TEST_P(TensorTestSuite, TensorBraceInitializationRank1Integer)
{
  typename TypeParam::Tensor tensor1{ 11, 22, 33, 44, 55, 66 };

  EXPECT_TRUE(grid::is_tensor_v<decltype(tensor1)>);
  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(6));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<int>(1)));

  int data[] = { 11, 22, 33, 44, 55, 66 };
  EXPECT_EQ(memcmp(tensor1.Data(), data, sizeof(data)), 0);
}

TYPED_TEST_P(TensorTestSuite, TensorBraceInitializationRank2Integer)
{
  typename TypeParam::Tensor tensor1{ { 11, 12 }, { 21, 22, 23 }, { 31, 32, 33, 34 } };

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(3, 4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<int>(4), size<int>(1)));

  const int* data = reinterpret_cast<const int*>(tensor1.Data());
  EXPECT_EQ(data[0], 11);
  EXPECT_EQ(data[4], 21);
  EXPECT_EQ(data[8], 31);
  EXPECT_EQ(data[9], 32);
}

TYPED_TEST_P(TensorTestSuite, TensorAllocInitializedRank1Double)
{
  typename TypeParam::Tensor tensor1(4UL, 1.2);

  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(1)));

  double verify[] = { 1.2, 1.2, 1.2, 1.2 };
  EXPECT_EQ(memcmp(tensor1.Data(), verify, sizeof(verify)), 0);
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedRank1Double)
{
  typename TypeParam::Tensor tensor1(5UL, grid::Uninitialized<double>{});
  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(5));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(1)));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocInitializedRank2Char)
{
  typename TypeParam::Tensor tensor1(5UL, 4UL, (char)'3');

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(5, 4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<char>(4), size<char>(1)));

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
  EXPECT_THAT(tensor1.Dims(), ElementsAre(7, 3));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<int>(3), size<int>(1)));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocInitializedRank3Double)
{
  typename TypeParam::Tensor tensor1{{4, 5, 7}, 3.3};

  EXPECT_EQ(tensor1.Rank(), 3);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(4, 5, 7));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(7 * 5), size<double>(7), size<double>(1)));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedRank3Double)
{
  typename TypeParam::Tensor t33({3, 2, 1}, grid::Uninitialized<double>{});
  EXPECT_THAT(t33.Strides(), ElementsAre(size<double>(2 * 1), size<double>(1), size<double>(1)));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedPattedRank3Double)
{
  typename TypeParam::Tensor tensor1({3, 2, 1}, {size<double>(2 * 2 * 4), size<double>(2 * 2), size<double>(2)}, grid::Uninitialized<double>{});
  EXPECT_EQ(tensor1.Rank(), 3);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(3, 2, 1));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(2 * 2 * 4), size<double>(2 * 2), size<double>(2)));
}


TYPED_TEST_P(TensorTestSuite, TensorMMap)
{
  std::FILE* tmpf = std::tmpfile();

  std::array<size_t, 4> ds1 = {2, 3, 2, 4};
  std::fwrite(ds1.data(), sizeof ds1[0], ds1.size(), tmpf);

  std::array<double, 4> row1 = {1.2, 2.3, 3.4, 0.0};
  std::fwrite(row1.data(), sizeof row1[0], row1.size(), tmpf);
  std::fwrite(row1.data(), sizeof row1[0], row1.size(), tmpf);

  std::array<size_t, 4> ds2 = {2, 3, 2, 3};
  std::fwrite(ds2.data(), sizeof ds2[0], ds2.size(), tmpf);

  std::array<double, 3> row2 = {4.3, 3.2, 2.1};
  std::fwrite(row2.data(), sizeof row2[0], row2.size(), tmpf);
  std::fwrite(row2.data(), sizeof row2[0], row2.size(), tmpf);

  size_t file_size = std::ftell(tmpf);
  EXPECT_EQ(file_size, sizeof(size_t) * 8 + sizeof(double) * (8 + 6));

  std::rewind(tmpf);

  int fd = fileno(tmpf);
  auto mmap = std::make_shared<grid::MMap>(fd, file_size);
  grid::MMapView view(mmap);

  auto dims1 = view.Read<std::array<size_t, 2>>();
  auto strides1 = view.Read<std::array<size_t, 2>>();
  typename TypeParam::Tensor tensor1(view.Array<double>(0UL, dims1, grid::make_strides<double>(strides1)));

  auto dims2 = view.Read<std::array<size_t, 2>>();
  auto strides2 = view.Read<std::array<size_t, 2>>();
  typename TypeParam::Tensor tensor2(view.Array<double>(0UL, dims2, grid::make_strides<double>(strides2)));

  std::fclose(tmpf);
}


REGISTER_TYPED_TEST_SUITE_P(TensorTestSuite,
    TensorBraceInitializationRank0Integer,
    TensorBraceInitializationRank1Integer,
    TensorBraceInitializationRank2Integer,
    TensorAllocInitializedRank1Double,
    TensorAllocUninitializedRank1Double,
    TensorAllocInitializedRank2Char,
    TensorAllocUninitializedRank2Double,
    TensorAllocInitializedRank3Double,
    TensorAllocUninitializedRank3Double,
    TensorAllocUninitializedPattedRank3Double,
    TensorMMap);