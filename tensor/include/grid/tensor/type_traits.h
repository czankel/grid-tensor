//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TYPE_TRAITS_H
#define GRID_TENSOR_TYPE_TRAITS_H

namespace grid {


// Handle 16-bit floating point support.
// Provide TypeID<type>  ... FIXME
using float16_t = __fp16;
static const char* float16_type_info_name_g = "fp16";
class float16_type_info : public std::type_info
{
 public:
  // TODO: implement type_info methods
  float16_type_info() : std::type_info(float16_type_info_name_g) {}
};
static const float16_type_info float16_type_info_g {};
template <typename> constexpr const std::type_info& TypeID() { return typeid(void);}
template <> constexpr const std::type_info& TypeID<float16_t&>() { return float16_type_info_g; }

// Define promotions to higher precision for each type
template <typename> struct promote;
template <> struct promote<float> { using type = double; };
template <> struct promote<float16_t>  { using type = float; };
template <> struct promote<int8_t> { using type = int16_t; };
template <typename T> using promote_t = typename promote<T>::type;

//
// Quantizations
//

struct BlockQ4_0
{
  constexpr static size_t size = 32;
  float16_t delta;
  uint8_t qs[size / 2];
};

struct BlockQ4_1
{
  constexpr static size_t size = 32;
  float16_t delta;
  float16_t min;
  uint8_t qs[size / 2];
};

struct BlockQ5_0
{
  constexpr static size_t size = 32;
  float16_t delta;
  uint8_t qh[4];          // 5th bit of quants
  uint8_t qs[size / 2];
};

struct BlockQ5_1 {
  constexpr static size_t size = 32;
  float16_t delta;
  float16_t min;
  uint8_t qh[4];          // 5th bit of quants
  uint8_t qs[size / 2];
};

struct BlockQ8_0
{
  constexpr static size_t size = 32;
  float16_t delta;
  uint8_t qs[size];
};

struct BlockQ8_1 {
  constexpr static size_t size = 32;
  float16_t delta;
  float16_t min;
  uint8_t qs[size];
};


} // end of namespace grid

#endif  // GRID_TENSOR_TYPE_TRAITS_H
