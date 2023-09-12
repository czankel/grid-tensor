//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_SLOWCPU_OPERATORS_H
#define GRID_TENSOR_SLOWCPU_OPERATORS_H

namespace grid {


namespace details {

template <typename _T, size_t>
inline std::enable_if_t<!std::is_floating_point_v<_T>, bool>
equals(const char* src1, const char* src2,
       std::span<const size_t,  1> dims,
       std::span<const ssize_t, 1> strides1,
       std::span<const ssize_t, 1> strides2)
{
  for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
    if (*reinterpret_cast<const _T*>(src1) != *reinterpret_cast<const _T*>(src2))
      return false;
  return true;
}

template <typename _T, size_t>
inline std::enable_if_t<std::is_floating_point_v<_T>, bool>
equals(const char* src1, const char* src2,
       std::span<const size_t,  1> dims,
       std::span<const ssize_t, 1> strides1,
       std::span<const ssize_t, 1> strides2)
{
  constexpr _T max_abs_error = std::numeric_limits<_T>::epsilon() * 100;

  for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
  {
    _T data0 = *reinterpret_cast<const _T*>(src1);
    _T data1 = *reinterpret_cast<const _T*>(src2);

    if (std::abs(data0 - data1) > max_abs_error)
      return false;
  }
  return true;
}

template <typename _T, size_t _N>
inline std::enable_if_t<(_N > 1), bool>
equals(const char* src1, const char* src2,
       std::span<const size_t,  _N> dims,
       std::span<const ssize_t, _N> strides1,
       std::span<const ssize_t, _N> strides2)
{
  static_assert(_N != std::dynamic_extent, "dynamic_extent not allowed");
  for (size_t i = 0; i < dims[0]; i++, src1 += strides1[0], src2 += strides2[0])
    if (!equals<_T, _N - 1>(src1, src2,
                            std::span<const size_t,  _N - 1>(dims.begin() + 1, _N - 1),
                            std::span<const ssize_t, _N - 1>(strides1.begin() + 1, _N - 1),
                            std::span<const ssize_t, _N - 1>(strides2.begin() + 1, _N - 1)))
      return false;

  return true;
}

} // end of namespace details


template <TensorFor<TensorSlowCpu> _Tensor1, TensorFor<TensorSlowCpu> _Tensor2>
bool operator==(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  constexpr size_t _Rank = tensor1.Rank();
  static_assert(_Rank == tensor2.Rank(), "ranks mismatch between tensors");

  return details::equals<typename std::remove_cvref_t<_Tensor1>::value_type, _Rank>(
                         reinterpret_cast<const char*>(tensor1.Data()),
                         reinterpret_cast<const char*>(tensor2.Data()),
                         std::span(tensor1.Dims()),
                         std::span(tensor1.Strides()),
                         std::span(tensor2.Strides()));
}


/// TensorAdd<TensorSlowCpu> implements tensor addition operation for tensors of the same rank.
template <typename _T, size_t _Rank, TensorFor<TensorSlowCpu> _Tensor1, TensorFor<TensorSlowCpu> _Tensor2>
struct TensorAdd<TensorSlowCpu, _T, _Rank, _Tensor1, _Tensor2> : TensorBaseOp //<_Tensor1>
{
  constexpr static size_t Rank()                  { return _Rank; }
  using tensor_type = TensorSlowCpu<_T, _Rank>;
  using value_type = _T;

  template <ConvertibleTensorFor<TensorSlowCpu> T1, ConvertibleTensorFor<TensorSlowCpu> T2>
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

  // Functor

  // Rank1 + Rank1
  template <TensorRank<1> = _Tensor1, TensorRank<1> = _Tensor2>
  auto operator()() const
  {
    size_t dim_m = tensor1_.Dims()[0];

    // FIXME: stride
    const value_type* data1 = reinterpret_cast<const value_type*>(tensor1_.Data());
    const value_type* data2 = reinterpret_cast<const value_type*>(tensor2_.Data());

    // FIXME: this would be a local variable???
    auto res = TensorSlowCpu(dim_m, Uninitialized<value_type>{});
    value_type* sum = reinterpret_cast<value_type*>(res.Data());

    for (size_t i = 0; i < dim_m; i++)
      sum[i] = data1[i] + data2[i];
    //auto res = TensorSlowCpu(0, Uninitialized<value_type>{});
    return res;
  }

  // Rank2 + Rank2
  template <TensorRank<2> = _Tensor1, TensorRank<2> = _Tensor2>
  auto operator()() const
  {
    size_t dim_m = tensor1_.Dims()[0];
    size_t dim_n = tensor1_.Dims()[1];

    const value_type* data1 = reinterpret_cast<const value_type*>(tensor1_.Data());
    const value_type* data2 = reinterpret_cast<const value_type*>(tensor2_.Data());

    auto res = TensorSlowCpu(dim_m, dim_n, Uninitialized<value_type>{});
    value_type* sum = reinterpret_cast<value_type*>(res.Data());

    // FIXME: handle stride
    for (size_t i = 0; i < dim_m; i++)
      for (size_t j = 0; j < dim_n; j++)
        sum[i * dim_n + j] = data1[i * dim_n + j] + data2[i * dim_n + j];
#if 0
    auto& tensor1 = std::get<0>(tensors_);
    size_t dim_m = tensor1.Dim(0);
    size_t dim_n = tensor1.Dim(1);
    auto res = TensorSlowCpu(dim_m, dim_n, Uninitialized<value_type>{});
#endif
    return res;
  }
#if 0
  // FIXME: stride
  // Rank3 + Rank3
  template <TensorRank<3> = _Tensor1, TensorRank<3> = _Tensor2>
  auto operator()() const
  {
    size_t dim_m = tensor1_.Dim(0);
    size_t dim_n = tensor1_.Dim(1);
    size_t dim_c = tensor1_.Dim(2);

    size_t stride1_m = tensor1_.Stride(0);
    size_t stride1_n = tensor1_.Stride(1);
    size_t stride1_c = tensor1_.Stride(2);
    size_t stride2_m = tensor2_.Stride(0);
    size_t stride2_n = tensor2_.Stride(1);
    size_t stride2_c = tensor2_.Stride(2);

    const value_type* data1 = tensor1_.Data();
    const value_type* data2 = tensor2_.Data();

    auto res = TensorSlowCpu(dim_m, dim_n, dim_c, Uninitialized<value_type>{});
    value_type* sum = res.Data();

    // FAST
    if (stride1_c == 1 && stride2_c == 1)
      for (size_t i = 0; i < dim_m; i++)
        for (size_t j = 0; j < dim_n; j++)
          for (size_t k = 0; j < dim_c; j++)
            sum[(i * dim_n + j) * dim_c + k] =
              data1[(i * dim_n + j) * dim_c + k] + data2[(i * dim_n + j) * dim_c + k];
    else
      for (size_t i = 0; i < dim_m; i++)
        for (size_t j = 0; j < dim_n; j++)
          for (size_t k = 0; j < dim_c; j++)
            sum[(i * dim_n + j) * dim_c + k] =
              data1[(i * dim_n + j) * dim_c + k] + data2[(i * dim_n + j) * dim_c + k];

    return res;
  }
#endif
  _Tensor1 tensor1_;
  _Tensor2 tensor2_;
};


// CTAD

template <ConvertibleTensorFor<TensorSlowCpu> _Tensor1, ConvertibleTensorFor<TensorSlowCpu> _Tensor2>
TensorAdd(_Tensor1, _Tensor2)
  -> TensorAdd<TensorSlowCpu, typename _Tensor2::value_type, _Tensor1::Rank(),
               typename _Tensor1::tensor_type, typename _Tensor2::tensor_type>;

} // end of namespace grid

#endif // GRID_TENSOR_SLOWCPU_OPERATORS_H
