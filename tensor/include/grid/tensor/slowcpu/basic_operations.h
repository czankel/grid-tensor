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

// Operator==
template <grid::TensorSlowCpuType _Tensor1, grid::TensorSlowCpuType _Tensor2>
bool operator==(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  using value_type = typename std::remove_cvref_t<_Tensor1>::value_type;
  if constexpr (!std::is_same_v<typename std::remove_cvref_t<_Tensor1>::value_type,
                                typename std::remove_cvref_t<_Tensor2>::value_type>)
    return false;

  constexpr value_type max_abs_error = std::numeric_limits<value_type>::epsilon() * 100;

  size_t rank = tensor1.Rank();
  if (rank != tensor2.Rank())
    return false;

  const unsigned int dim0 = rank > 0 ? tensor1.Dim(0) : 1;
  const unsigned int dim1 = rank > 1 ? tensor1.Dim(1) : 1;
  const unsigned int dim2 = rank > 2 ? tensor1.Dim(2) : 1;
  const unsigned int dim3 = rank > 3 ? tensor1.Dim(3) : 1;

  if ((rank > 0 && tensor2.Dim(0) != dim0) ||
      (rank > 1 && tensor2.Dim(1) != dim1) ||
      (rank > 2 && tensor2.Dim(2) != dim2) ||
      (rank > 3 && tensor2.Dim(3) != dim3) ||
       rank > 4)
    return false;

  const unsigned int stride1_0 = rank > 0 ? tensor1.Stride(0) : 1;
  const unsigned int stride1_1 = rank > 1 ? tensor1.Stride(1) : 1;
  const unsigned int stride1_2 = rank > 2 ? tensor1.Stride(2) : 1;
  const unsigned int stride2_0 = rank > 0 ? tensor2.Stride(0) : 1;
  const unsigned int stride2_1 = rank > 1 ? tensor2.Stride(1) : 1;
  const unsigned int stride2_2 = rank > 2 ? tensor2.Stride(2) : 1;

  const value_type* data0 = tensor1.Data();
  const value_type* data1 = tensor2.Data();

  for (unsigned int c = 0; c < dim3; c++)
    for (unsigned int k = 0; k < dim2; k++)
      for (unsigned int m = 0; m < dim1; m++)
        for (unsigned int n = 0; n < dim0; n++)
          if (std::abs(data0[((((c * stride1_2) + k) * stride1_1 + m) * stride1_0 + n)] -
                       data1[(((c * stride2_2) + k) * stride2_1 + m) * stride2_0]) > max_abs_error)
            return  false;

  return true;
}


/// TensorAdd<TensorSlowCpu> implements tensor addition operation for tensors of the same rank.
template <size_t _Rank, typename _T, TensorSlowCpuType _Tensor1, TensorSlowCpuType _Tensor2>
struct TensorAdd<TensorSlowCpu, _Rank, _T, _Tensor1, _Tensor2> : TensorBaseOp
{
  constexpr static size_t Rank()                  { return _Rank; }
  using value_type = _T;

  template <typename T1, typename T2>
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
  template <TensorR1Type = _Tensor1, TensorR1Type = _Tensor2>
  auto operator()() const
  {
    size_t dim_m = tensor1_.Dim(0);

    // FIXME: stride
    const value_type* data1 = tensor1_.Data();
    const value_type* data2 = tensor2_.Data();

    // FIXME: this would be a local variable???
    auto res = TensorSlowCpu(dim_m, Uninitialized<value_type>{});
    value_type* sum = res.Data();

    for (size_t i = 0; i < dim_m; i++)
      sum[i] = data1[i] + data2[i];
    //auto res = TensorSlowCpu(0, Uninitialized<value_type>{});
    return res;
  }

  // Rank2 + Rank2
  template <TensorR2Type = _Tensor1, TensorR2Type = _Tensor2>
  auto operator()() const
  {
    size_t dim_m = tensor1_.Dim(0);
    size_t dim_n = tensor1_.Dim(1);

    const value_type* data1 = tensor1_.Data();
    const value_type* data2 = tensor2_.Data();

    auto res = TensorSlowCpu((unsigned int)dim_m, (unsigned int)dim_n, Uninitialized<value_type>{});
    value_type* sum = res.Data();

    // FIXME: handle stride
    for (size_t i = 0; i < dim_m; i++)
      for (size_t j = 0; j < dim_n; j++) {
        sum[i * dim_n + j] = data1[i * dim_n + j] + data2[i * dim_n + j];
      }
#if 0
    auto& tensor1 = std::get<0>(tensors_);
    size_t dim_m = tensor1.Dim(0);
    size_t dim_n = tensor1.Dim(1);
    auto res = TensorSlowCpu((unsigned int)dim_m, (unsigned int)dim_n, Uninitialized<value_type>{});
#endif
    return res;
  }
#if 0
  // FIXME: stride
  // Rank3 + Rank3
  template <TensorR3Type = _Tensor1, TensorR3Type = _Tensor2>
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

template <TensorSlowCpuType _Tensor1, TensorSlowCpuType _Tensor2>
TensorAdd(_Tensor1, _Tensor2)
  -> TensorAdd<TensorSlowCpu, _Tensor1::Rank(), typename _Tensor1::value_type, _Tensor1, _Tensor2>;


} // end of namespace grid

#endif // GRID_TENSOR_SLOWCPU_OPERATORS_H
