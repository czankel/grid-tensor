//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_PARAMETERS_H
#define GRID_TENSOR_TENSOR_PARAMETERS_H

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
template <typename T, size_t TRank, typename Indices = std::make_index_sequence<TRank>>
std::array<ssize_t, TRank> make_strides(const std::array<size_t, TRank>& dimensions)
{
  std::array<ssize_t, TRank> strides;
  ssize_t stride = sizeof(T);
  for (int i = static_cast<int>(TRank) - 1; i >= 0; i--)
  {
    strides[i] = dimensions[i] != 1 ? stride : 0;
    stride *= dimensions[i];
  }
  return strides;
}

// get_buffer_size returns the size of the buffer from dimensions and strides.
template<size_t TRank>
size_t get_buffer_size(const std::array<size_t, TRank>& dimensions, const std::array<ssize_t, TRank>& strides)
{
  size_t size = 0;
  for (size_t i = 0; i < TRank; i++)
    size = std::max(size, dimensions[i] * strides[i]);
  return size;
}

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_PARAMETERS_H
