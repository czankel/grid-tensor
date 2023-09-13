//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor_slowcpu.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ElementsAre;

namespace {
template <typename _T> constexpr size_t size(size_t count) { return sizeof(_T) * count; }
}

template <typename _T, size_t _Rank, auto... _Args>
using Tensor = grid::TensorSlowCpu<_T, _Rank, _Args...>;

TEST(TensorSlowCPU, TensorBaseInitializationRank0Integer)
{
  Tensor tensor{ 4 };
  EXPECT_EQ(tensor.Rank(), 0);
  EXPECT_THAT(tensor.Dims(), ElementsAre());
  EXPECT_THAT(tensor.Strides(), ElementsAre());
}

TEST(TensorSlowCPU, TensorBraceInitializationRank1Integer)
{
  Tensor tensor1{ 11, 22, 33, 44, 55, 66 };

  EXPECT_TRUE(grid::is_tensor_v<decltype(tensor1)>);
  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(6));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<int>(1)));

  int data[] = { 11, 22, 33, 44, 55, 66 };
  EXPECT_EQ(memcmp(tensor1.Data(), data, sizeof(data)), 0);
}

TEST(TensorSlowCPU, TensorBraceInitializationRank2Integer)
{
  Tensor tensor1{ { 11, 12 }, { 21, 22, 23 }, { 31, 32, 33, 34 } };

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(3, 4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<int>(4), size<int>(1)));

  const int* data = reinterpret_cast<const int*>(tensor1.Data());
  EXPECT_EQ(data[0], 11);
  EXPECT_EQ(data[4], 21);
  EXPECT_EQ(data[8], 31);
  EXPECT_EQ(data[9], 32);
}

TEST(TensorSlowCPU, TensorAllocInitializedRank1Double)
{
  Tensor tensor1(4UL, 1.2);

  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(1)));

  double verify[] = { 1.2, 1.2, 1.2, 1.2 };
  EXPECT_EQ(memcmp(tensor1.Data(), verify, sizeof(verify)), 0);
}

TEST(TensorSlowCPU, TensorAllocUninitializedRank1Double)
{
  Tensor tensor1(5UL, grid::Uninitialized<double>{});
  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(5));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(1)));
}

TEST(TensorSlowCPU, TensorAllocInitializedRank2Char)
{
  Tensor tensor1(5UL, 4UL, (char)'3');

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(5, 4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<char>(4), size<char>(1)));

  char verify[] = { '3', '3', '3', '3', '3',
                    '3', '3', '3', '3', '3',
                    '3', '3', '3', '3', '3',
                    '3', '3', '3', '3', '3' };
  EXPECT_EQ(memcmp(tensor1.Data(), verify, sizeof(verify)), 0);
}

TEST(TensorSlowCPU, TensorAllocUninitializedRank2Double)
{
  Tensor tensor1(7UL, 3UL, grid::Uninitialized<int>{});

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(7, 3));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<int>(3), size<int>(1)));
}

TEST(TensorSlowCPU, TensorAllocInitializedRank3Double)
{
  Tensor tensor1{{4, 5, 7}, 3.3};

  EXPECT_EQ(tensor1.Rank(), 3);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(4, 5, 7));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(7 * 5), size<double>(7), size<double>(1)));
}

TEST(TensorSlowCPU, TensorAllocUninitializedRank3Double)
{
  Tensor t33({3, 2, 1}, grid::Uninitialized<double>{});
  EXPECT_THAT(t33.Strides(), ElementsAre(size<double>(2 * 1), size<double>(1), size<double>(1)));
}


TEST(TensorSlowCPU, TensorAllocUninitializedPattedRank3Double)
{
  Tensor tensor1({3, 2, 1}, {size<double>(2 * 2 * 4), size<double>(2 * 2), size<double>(2)}, grid::Uninitialized<double>{});
  EXPECT_EQ(tensor1.Rank(), 3);
  EXPECT_THAT(tensor1.Dims(), ElementsAre(3, 2, 1));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(size<double>(2 * 2 * 4), size<double>(2 * 2), size<double>(2)));
}


TEST(TensorSlowCpu, TensorMMap)
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
  Tensor tensor1(view.Array<double>(0UL, dims1, grid::make_strides<double>(strides1)));

  auto dims2 = view.Read<std::array<size_t, 2>>();
  auto strides2 = view.Read<std::array<size_t, 2>>();
  Tensor tensor2(view.Array<double>(0UL, dims2, grid::make_strides<double>(strides2)));

  auto res = (tensor1 + tensor2)();
  Tensor v{ { 5.5, 5.5, 5.5 }, { 5.5, 5.5, 5.5 } };
  EXPECT_EQ(res, v);

  std::fclose(tmpf);
}
