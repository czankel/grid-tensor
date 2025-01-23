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

#include "concepts.h"

namespace grid {

// get_array(iniitlizer_list)
// returns a std::array initialized from a initializer list.
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

// get_array(initializer_list<initializer_list>)
// returns a std::array from a 2-dimensional initializer list.
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

// get_array(initializer_list<initializer_list<initializer_list>>)
// returns a std::array from a 3-dimensional initializer list.
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


// get_array(T(&)[])
// returns a std::array from a c-array.
template <Arithmetic T, size_t N>
inline constexpr std::array<T, N>
get_array(const T(&init)[N])
{
  std::array<T, N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// get_array(T(&&)[])
// returns a std::array from a c-array (rvalue reference)
template <Arithmetic T, size_t N>
inline constexpr std::array<T, N>
get_array(T(&&init)[N])
{
  std::array<T, N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// get_array(T(&&...)[N])
// returns a std::array from a 2-dimensional c-array
template <Arithmetic T, size_t... N>
inline constexpr std::array<T, sizeof...(N) * std::max({N...})>
get_array(T(&&... init)[N])
{
  constexpr size_t maxcols = std::max({N...});
  std::array<T, sizeof...(N) * maxcols> arr{};
  auto line_it = arr.begin();

  auto apply = [&] <typename U> (U&& value, size_t cols) -> void {
    for (size_t i = 0; i < cols; i++, ++line_it)
      *line_it = value[i];
    line_it += maxcols - cols;
  };

  (apply(std::forward<T[N]>(init), N),...);
  return arr;
}

// get_array(T((&&...)[M])[N])
// returns a std::array from a 3-dimensional c-array
template <Arithmetic T, size_t... M, size_t... N>
inline constexpr std::array<T, sizeof...(M) * std::max({M...}) * std::max({N...})>
get_array(T((&&... init)[M])[N])
{
  constexpr size_t maxrows = std::max({M...});
  constexpr size_t maxcols = std::max({N...});
  std::array<T, sizeof...(M) * maxrows * maxcols> arr{};
  auto line_it = arr.begin();

  auto apply = [&] <typename U> (U&& value, size_t rows, size_t cols) -> void {
    for (size_t i = 0; i < rows; i++, line_it += maxcols - cols)
      for (size_t j = 0; j < cols; j++, ++line_it)
        *line_it = value[i][j];
    line_it += maxrows - rows;
  };

  (apply(std::forward<T[M][N]>(init), M, N),...);
  return arr;
}


// make_strides returns a std::array with the strides calculated from the provided dimensions and
// the template type parameter (make_strides<TYPE>(...))
// FIXME: Broadcasting rules...
    // FIXME: required for mul(mat,vec) to apply vector for each col (broadcast)...
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
  size_t size = 1;  // default is rank-0, which has size 1
  auto di = std::forward<U>(dimensions).begin();
  auto si = std::forward<V>(strides).begin();
  for (; di != dimensions.end() && si != strides.end(); ++di, ++si)
    size = std::max(size, *di * *si);
  return size * sizeof(T);
}


// @brief Calculate the size of an (sub-)area with the given dimensions within a larger tensor.
template <typename T, typename U, typename V>
size_t get_block_size(U&& dimensions, V&& strides)
{
  size_t size = 1;  // default is rank-0, which has size 1
  auto di = std::forward<U>(dimensions).begin();
  auto si = std::forward<V>(strides).begin();
  for (; di != dimensions.end() && si != strides.end(); ++di, ++si)
    size += std::max(*di - 1, 0UL) * *si;
  return size * sizeof(T);
}


// Broadcast expands dimensions ("broadcasting") of the left tensor to match the right tensor
template <typename TTensor1, typename TTensor2>
inline auto BroadcastDimensions(const TTensor1& tensor1, const TTensor2& tensor2)
{
  constexpr size_t rank1 = TTensor1::rank;
  constexpr size_t rank2 = TTensor2::rank;
  constexpr size_t drank1 = rank2 > rank1 ? rank2 - rank1 : 0UL;
  constexpr size_t drank2 = rank1 > rank2 ? rank1 - rank2 : 0UL;

  if constexpr (rank1 == 0)
    return tensor2.Dimensions();
  else if constexpr (rank2 == 0)
    return tensor1.Dimensions();
  else
  {
    const auto& dimensions1 = tensor1.Dimensions();
    const auto& dimensions2 = tensor2.Dimensions();

    std::array<size_t, std::max(rank1, rank2)> dimensions;
    std::generate(dimensions.begin(), dimensions.end(), [n = 0, &dimensions1, &dimensions2]() mutable -> size_t
    {
      size_t k = n++;
      if (k < drank1 || dimensions1[k - drank1] == 1)
        return dimensions2[k - drank2];
      else if (k < drank2 || dimensions2[k - drank2] == 1 || dimensions2[k - drank2] == dimensions1[k - drank1])
        return dimensions1[k - drank1];
      else
        throw std::runtime_error("broadcast failed");
    });
    return dimensions;
  }
}

/// @brief Helper function to align strides returning an array if a stride needs to be extended.
template <size_t S1, size_t S2>
inline auto BroadcastStrides(std::span<const ssize_t, S1> strides1, std::span<const ssize_t, S2> strides2)
{
  if constexpr (S1 == S2)
    return std::make_tuple(std::move(strides1), std::move(strides2));
  else if constexpr (S1 == 0)
    return std::make_tuple(std::move(std::array<const ssize_t, S2>{}), std::move(strides2));
  else if constexpr (S2 == 0)
    return std::make_tuple(std::move(strides1), std::move(std::array<const ssize_t, S1>{}));
  else if constexpr (S2 > S1)
  {
    std::array<ssize_t, S2> strides{};
    std::ranges::copy(strides1, strides.begin() + S2 - S1);
    return std::make_tuple(std::move(strides), std::move(strides2));
  }
  else
  {
    std::array<ssize_t, S1> strides{};
    std::ranges::copy(strides2, strides.begin() + S1 - S2);
    return std::make_tuple(std::move(strides1), std::move(strides));
  }
}

// TODO: the CUDA nvcc compiler doesn't support trailing return types
#if !defined(__CUDACC__)

/// @brief Helper function to reduce the rank in case of contiguous data for binary operators.
///
/// The operator function is called with a boolean flag to indicate that the "lowest" dimension
/// is contiguous (strides are all implicit 1), and a new dimension span for the remaining
/// dimension or non-contiguous dimensions.
///
/// Folding rules:
///
///  dim x,y strides 0,0  -> fold (scalar applied to y not z) -> dim x*y Note: expected stride must reset to 1
///  dim x,y strides 1,0  -> no-fold (col-vector applied to each col x)
///  dim x,y strides y,0  -> no-fold (col-vector applied to each col x)
///  dim x,y strides 0,1  -> no-fold (row-vector applied to each row y)
///  dim x,y strides y,1  -> fold (stride is y) -> dim x*y
///  dim x,y strides _,1  -> no-fold (stride is not y)
///  dim x,y strides by,b -> fold
///  dim x,y strides _,b  -> no-fold (stride is not b*y)
///
///  dim _,1 strides _,_ -> fold (dim is 1, last stride is never applied) -> dim _,  contiguous depends on next iteration
///  dim 1,_ strides _,_ -> non-fold, would require to change dimensions and strides
///
///  contiguous means that the returning dimensions (may be unchanged) are contiguous or stride-0.
///
/// FIXME: if dimension[x] == 1, stride 0 or 1 is the same
///
template <size_t TRank, typename TOp>
void Fold(TOp&& op, std::span<const size_t, TRank> dimensions, auto... strides)
{
  // rank-0: scalars return 'contiguous'
  if constexpr (TRank == 0)
    op(std::move(dimensions), true);

  // rank-1: vectors cannot be folded; contiguous if single-element vector or strides are 0 or 1
  else if constexpr (TRank == 1)
    op(std::move(dimensions), dimensions[0] == 1 || ((strides.size() > 0 && strides[0] <= 1) && ...));

  // rank > 1: cannot fold if strides are not 0 or 1 (unless scalar: no stride)
  else if (((strides.size() > 0 && strides[strides.size() - 1] > 1) || ...))
    op(std::move(dimensions), false);

  // FIXME: all strides must be 0 to be a scalar, if any is not 0, not a scalar, cannot fold (what about 1)??

  // try to fold
  else
  {
    // FIXME: does this require aligned strides or explicitly not aligned?
    // FIXME: isn't contiguous OP 0-strides also contiguous?
    printf("fold check\n");
    static_assert(sizeof...(strides) > 1);
    constexpr size_t max_folds = std::min({strides.size()...});
    size_t fold_dim = dimensions[TRank - 1];

    auto foldfn = [&]<size_t I>() -> bool
    {
      // strides.size() > I - 2 &&
      if (I == max_folds - 1 ||
          (((strides.size() > I - 1) && (strides[strides.size() - I - 2] - fold_dim != 0)) || ...))
      {
        std::array<size_t, TRank - I> dim;
        std::ranges::copy(dimensions.template first<TRank - I - 1>(), dim.begin());
        dim[TRank - I - 1] = fold_dim;
        op(std::span<const size_t, TRank - I>(dim.begin(), TRank - I), true);
        return false;
      }
      fold_dim *= dimensions[TRank - I - 2];
      return true;
    };

    [&] <std::size_t... I>(std::index_sequence<I...>)
    {
      (foldfn.template operator()<I>() && ...);
    }(std::make_index_sequence<max_folds>{});
  }
}

#endif // !defined(__CUDACC__)

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_PARAMETERS_H
