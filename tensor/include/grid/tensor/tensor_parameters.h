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
template <typename _Tp, size_t... _Ns>
inline constexpr std::array<_Tp, sizeof...(_Ns)>
get_array_impl(std::initializer_list<_Tp>&& init, std::index_sequence<_Ns...>)
{
  return std::array<_Tp, sizeof...(_Ns)>{ *(init.begin() + _Ns) ... };
}

template <typename _Tp, size_t _N, typename _Ns = std::make_index_sequence<_N>>
inline constexpr std::array<_Tp, _N>
get_array(std::initializer_list<_Tp>&& init)
{
  return get_array_impl(std::move(init), _Ns{});
}

// get_array(initializer_list<initializer_list>) returns a std::array from a 2-dimensional initializer list.
template <typename _Tp, size_t _M, size_t _N>
inline constexpr std::array<_Tp, _M * _N>
get_array(std::initializer_list<std::initializer_list<_Tp>>&& init)
{
  std::array<_Tp, _M * _N> arr{};
  auto line_it = arr.begin();
  for (auto it : init)
  {
    std::copy(it.begin(), it.end(), line_it);
    line_it += _N;
  }
  return arr;
}

// get_array(initializer_list<initializer_list<initializer_list>>) returns a std::array from a
// 3-dimensional initializer list.
template <typename _Tp, size_t _C, size_t _M, size_t _N>
inline constexpr std::array<_Tp, _C * _M * _N>
get_array(std::initializer_list<std::initializer_list<std::initializer_list<_Tp>>>&& init)
{
  std::array<_Tp, _C * _M * _N> arr{};
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

// get_array(T(&)[]) returns a std::array from a c-array.
template <typename _Tp, size_t _N>
inline constexpr std::array<_Tp, _N>
get_array(const _Tp(&init)[_N])
{
  std::array<_Tp, _N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// get_array(T(&&)[]) returns a std::array from a c-array (rvalue reference)
template <typename _Tp, size_t _N>
inline constexpr std::array<_Tp, _N>
get_array(_Tp(&&init)[_N])
{
  std::array<_Tp, _N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// make_strides returns a std::array with the strides calculated from the provided dimensions and
// the template type parameter (make_strides<TYPE>(...))
template <typename _Tp, size_t _Rank, size_t... Is>
constexpr std::array<ssize_t, _Rank>
make_strides_impl(const std::array<size_t, _Rank>& dimensions, std::index_sequence<Is...>)
{
  auto multiply = [&dimensions](size_t index) {
    ssize_t res = sizeof(_Tp);
    for (size_t i = 0; i < _Rank - 1 - index; i++)
      res *= dimensions[_Rank - 1 - i];
    return res;
  };
  return std::array<ssize_t, _Rank>{multiply(Is)...};
}

template <typename _Tp, size_t _Rank, typename Indices = std::make_index_sequence<_Rank>>
std::array<ssize_t, _Rank> make_strides(const std::array<size_t, _Rank>& dimensions)
{
  return make_strides_impl<_Tp>(dimensions, Indices{});
}

// get_buffer_size returns the size of the buffer from dimensions and strides.
template<size_t _Rank>
size_t get_buffer_size(const std::array<size_t, _Rank>& dimensions, const std::array<ssize_t, _Rank>& strides)
{
  size_t size = 0;
  for (size_t i = 0; i < _Rank; i++)
    size = std::max(size, dimensions[i] * strides[i]);
  return size;
}

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_PARAMETERS_H
