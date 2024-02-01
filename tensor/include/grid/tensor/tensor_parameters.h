//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_PARAMETERS_H
#define GRID_TENSOR_TENSOR_PARAMETERS_H

#include <algorithm>

namespace grid {

// get_array(iniitlizer_list) returns a std::array initialized from a brace-initializer list.
template <typename T, size_t... Ns>
inline constexpr std::array<T, sizeof...(Ns)>
get_array_impl(std::initializer_list<T>&& init, std::index_sequence<Ns...>)
{
  return std::array<T, sizeof...(Ns)>{ *(init.begin() + Ns) ... };
}

template <typename T, size_t N, typename Ns = std::make_index_sequence<N>>
inline constexpr std::array<T, N>
get_array(std::initializer_list<T>&& init)
{
  return get_array_impl(std::move(init), Ns{});
}

// get_array(initializer_list<initializer_list>) returns a std::array from a 2-dimensional initializer list.
template <typename T, size_t M, size_t N>
inline constexpr std::array<T, M * N>
get_array(std::initializer_list<std::initializer_list<T>>&& init)
{
  std::array<T, M * N> arr{};
  auto line_it = arr.begin();
  for (auto it : init)
  {
    std::copy(it.begin(), it.end(), line_it);
    line_it += N;
  }
  return arr;
}

// get_array(initializer_list<initializer_list<initializer_list>>) returns a std::array from a
// 3-dimensional initializer list.
template <typename T, size_t C, size_t M, size_t N>
inline constexpr std::array<T, C * M * N>
get_array(std::initializer_list<std::initializer_list<std::initializer_list<T>>>&& init)
{
  std::array<T, C * M * N> arr{};
  auto line_it = arr.begin();
  for (auto lt : init)
  {
    for (auto it : lt)
    {
      std::copy(it.begin(), it.end(), line_it);
      line_it += N;
    }
  }
  return arr;
}

// get_array(T(&)[]) returns a std::array from a c-array.
template <typename T, size_t N>
inline constexpr std::array<T, N>
get_array(const T(&init)[N])
{
  std::array<T, N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// get_array(T(&&)[]) returns a std::array from a c-array (rvalue reference)
template <typename T, size_t N>
inline constexpr std::array<T, N>
get_array(T(&&init)[N])
{
  std::array<T, N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// make_strides returns a std::array with the strides calculated from the provided dimensions and
// the template type parameter (make_strides<TYPE>(...))
template <size_t TRank, typename Indices = std::make_index_sequence<TRank>>
std::array<ssize_t, TRank> make_strides(const std::array<size_t, TRank>& dimensions)
{
  std::array<ssize_t, TRank> strides;
  ssize_t stride = 1;
  for (int i = static_cast<int>(TRank) - 1; i >= 0; i--)
  {
    strides[i] = dimensions[i] != 1 ? stride : 0;
    stride *= dimensions[i];
  }
  return strides;
}

// get_buffer_size returns the required size for the given dimensions and strides.
template <typename T, typename U, typename V>
size_t get_buffer_size(U&& dimensions, V&& strides)
{
  size_t size = 0;
  auto di = std::forward<U>(dimensions).begin();
  auto si = std::forward<V>(strides).begin();
  for (; di != dimensions.end() && si != strides.end(); ++di, ++si)
    size = std::max(size, *di * *si);
  return size * sizeof(T);
}

// Broadcast expands dimensions ("broadcasting") of the tensors to make them the same rank.
template <typename T1, typename T2, size_t Rank>
inline auto BroadcastDimensions(std::array<size_t, Rank>& dimensions,
                                const T1& tensor1,
                                const T2& tensor2)
{
  constexpr int delta = T1::rank - T2::rank;
  const auto& dimensions1 = tensor1.Dimensions();
  const auto& dimensions2 = tensor2.Dimensions();

  std::generate(dimensions.begin(), dimensions.end(), [n = 0, &dimensions1, &dimensions2]() mutable -> size_t
  {
    int k = n++;
    if (k < delta || dimensions2[k-delta] == 1)
      return dimensions1[k];
    else if (dimensions1[k] == 1 || dimensions1[k] == dimensions2[k-delta])
      return dimensions2[k-delta];
    else
      throw std::runtime_error("broadcast failed");
  });
}

template <typename TTensor1, typename TTensor2>
inline auto Broadcast(const TTensor1& tensor1, const TTensor2& tensor2)
{
  constexpr size_t rank = std::max(TTensor1::rank, TTensor2::rank);
  std::array<size_t, rank> dimensions;

  constexpr int delta = static_cast<int>(TTensor1::rank) - static_cast<int>(TTensor2::rank);
  if constexpr (delta == 0)
  {
    BroadcastDimensions(dimensions, tensor1, tensor2);
    return std::make_tuple(dimensions, std::cref(tensor1.Strides()), std::cref(tensor2.Strides()));
  }
  else if constexpr (delta > 0)
  {
    const auto& strides2 = tensor2.Strides();
    std::array<ssize_t, rank> strides{0};
    std::copy(strides2.begin(), strides2.end(), strides.begin() + delta);
    BroadcastDimensions(dimensions, tensor1, tensor2);
    return std::make_tuple(dimensions, std::cref(tensor1.Strides()), strides);
  }
  else
  {
    const auto& strides1 = tensor1.Strides();
    std::array<ssize_t, rank> strides{0};
    std::copy(strides1.begin(), strides1.end(), strides.begin() + (-delta));
    BroadcastDimensions(dimensions, tensor2, tensor1);
    return std::make_tuple(dimensions, strides, std::cref(tensor2.Strides()));
  }
}

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_PARAMETERS_H
