//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_ADDITION_H
#define GRID_TENSOR_BASE_ADDITION_H

namespace grid {

/// TensorAdd<Tensor> implements tensor addition operation for tensors of the same rank.
///
/// "Broadcasting" supports specfic additions, such as:
///
///   Matrix + column-vector --> Matrix is split into column blocks, each added with the vector
///   Matrix + row-vector    --> Matrix is split into row blocks, each added with the vector
///
/// Note that tensor ranks must always match. In the case of broadcasting, the dimension is simply set to 1.
template <typename T, size_t TRank, PrimitiveTensor TTensor1, PrimitiveTensor TTensor2>
requires (TTensor1::Rank() == TTensor2::Rank())
class TensorAdd<Tensor, T, TRank, TTensor1, TTensor2>
{
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  constexpr static size_t rank = TRank;

  template <ConvertibleTo<Tensor> T1, ConvertibleTo<Tensor> T2>
  TensorAdd(T1&& tensor1, T2&& tensor2)
   : tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {}

  ~TensorAdd() {}

  // delete assignment and copy/move constructors
  TensorAdd() = delete;
  TensorAdd(const TensorAdd& other) = delete;
  TensorAdd(TensorAdd&& other) = delete;
  TensorAdd& operator=(const TensorAdd& other) = delete;
  TensorAdd& operator=(TensorAdd&& other) = delete;

 private:
  // ExpandDimensions expands "broadcast" dimension (dim = 1) to the dimension of the other tensor.
  std::array<size_t, TRank> ExpandDimensions(const TTensor1& tensor1, const TTensor2& tensor2) const
  {
    const auto& dimensions1 = tensor1.Dimensions();
    const auto& dimensions2 = tensor2.Dimensions();

    std::array<size_t, TRank> dimensions;
    for (size_t i = 0; i < TRank; i++)
    {
      if (dimensions1[i] != dimensions2[i] && dimensions1[i] != 1 && dimensions2[i] != 1)
        throw std::runtime_error("Invalid dimensions");
      dimensions[i] = dimensions1[i] == 1 ? dimensions2[i] : dimensions1[i];
    }

    return dimensions;
  }

  inline void add(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  0> dimensions,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>,
                  std::span<const ssize_t, 0>) const
  {
    *dest = *src1 + *src2;
  }

  inline void add(pointer dest, const_pointer src1, const_pointer src2,
                  std::span<const size_t,  1> dimensions,
                  std::span<const ssize_t, 1>,
                  std::span<const ssize_t, 1> strides1,
                  std::span<const ssize_t, 1> strides2) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      dest[i] = *src1 + *src2;
      reinterpret_cast<const char*&>(src1) += strides1[0];
      reinterpret_cast<const char*&>(src2) += strides2[0];
    }
  }

  template <size_t N> inline
  void add(pointer dest, const_pointer src1, const_pointer src2,
           std::span<const size_t,  N> dimensions,
           std::span<const ssize_t, N> strides0,
           std::span<const ssize_t, N> strides1,
           std::span<const ssize_t, N> strides2) const
  {
    static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      add(dest, src1, src2,
          std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides0.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1),
          std::span<const ssize_t, N - 1>(strides2.begin() + 1, N - 1));

      reinterpret_cast<char*&>(dest) += strides0[0];
      reinterpret_cast<const char*&>(src1) += strides1[0];
      reinterpret_cast<const char*&>(src2) += strides2[0];
    }
  }
 public:

  /// operator()() executes the operation and returns a tensor.
  auto operator()() const
  {
    auto dimensions = ExpandDimensions(tensor1_, tensor2_);
    auto result = Tensor(dimensions, Uninitialized<value_type>{});

    add(result.Data(),
        tensor1_.Data(),
        tensor2_.Data(),
        std::span<const size_t, TRank>(dimensions),
        std::span<const ssize_t, TRank>(result.Strides()),
        std::span<const ssize_t, TRank>(tensor1_.Strides()),
        std::span<const ssize_t, TRank>(tensor2_.Strides()));
    return result;
  }

 private:
  TTensor1 tensor1_;
  TTensor2 tensor2_;
};

//
// CTAD
//

template <ConvertibleTo<Tensor> TTensor1, ConvertibleTo<Tensor> TTensor2>
TensorAdd(TTensor1, TTensor2)
  -> TensorAdd<Tensor, typename TTensor1::value_type, TTensor1::rank,
               typename to_tensor<TTensor1>::type, typename to_tensor<TTensor2>::type>;

} // end of namespace grid

#endif // GRID_TENSOR_BASE_ADDITION_H
