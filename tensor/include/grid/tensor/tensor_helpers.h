//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_HELPERS_H
#define GRID_TENSOR_TENSOR_HELPERS_H

namespace grid {

// helper function to get an array from a brace-initializer list.
template <typename _T, size_t... _Ns>
inline constexpr std::array<_T, sizeof...(_Ns)>
get_array_impl(std::initializer_list<_T>&& init, std::index_sequence<_Ns...>)
{
  return std::array<_T, sizeof...(_Ns)>{ *(init.begin() + _Ns) ... };
}

template <typename _T, size_t _N, typename _Ns = std::make_index_sequence<_N>>
inline constexpr std::array<_T, _N>
get_array(std::initializer_list<_T>&& init)
{
  return get_array_impl(std::move(init), _Ns{});
}

// helper function to return an array from a two-dimensional initializer list
template <typename _T, size_t _M, size_t _N>
inline constexpr std::array<_T, _M * _N>
get_array(std::initializer_list<std::initializer_list<_T>>&& init)
{
  std::array<_T, _M * _N> arr{};
  auto line_it = arr.begin();
  for (auto it : init)
  {
    std::copy(it.begin(), it.end(), line_it);
    line_it += _N;
  }
  return arr;
}

// helper function to return an array from a three-dimensional initializer list
template <typename _T, size_t _C, size_t _M, size_t _N>
inline constexpr std::array<_T, _C * _M * _N>
get_array(std::initializer_list<std::initializer_list<std::initializer_list<_T>>>&& init)
{
  std::array<_T, _C * _M * _N> arr{};
  auto line_it = arr.begin();
  for (auto lt : init)
  {
    for (auto it : lt)
    {
      std::copy(it.begin(), it.end(), line_it);
      line_it += _N;
    }
  }
  return arr;
}

// helper function to re turn an array from a c-array.
template <typename _T, size_t _N>
inline constexpr std::array<_T, _N>
get_array(const _T(&init)[_N])
{
  std::array<_T, _N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// helper function to re turn an array from an rvalue c-array.
template <typename _T, size_t _N>
inline constexpr std::array<_T, _N>
get_array(_T(&&init)[_N])
{
  std::array<_T, _N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// helper function to return the strides from dimensions. Use: make_strides<TYPE>(std::array)
template <typename _T, size_t _Rank, size_t... Is>
constexpr std::array<ssize_t, _Rank>
make_strides_impl(const std::array<size_t, _Rank>& dims, std::index_sequence<Is...>)
{
  auto multiply = [&dims](size_t index) {
    ssize_t res = sizeof(_T);
    for (size_t i = 0; i < _Rank - 1 - index; i++)
      res *= dims[_Rank - 1 - i];
    return res;
  };
  return std::array<ssize_t, _Rank>{multiply(Is)...};
}

template <typename _T, size_t _Rank, typename Indices = std::make_index_sequence<_Rank>>
std::array<ssize_t, _Rank> make_strides(const std::array<size_t, _Rank>& dims)
{
  return make_strides_impl<_T>(dims, Indices{});
}

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_HELPERS_H

