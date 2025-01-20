//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_PRECISION_H
#define GRID_TENSOR_PRECISION_H

// TODO: Placeholder

template <typename> struct Eps {};
template <> struct Eps<float>  { constexpr static float  default_value = 1e-5f; float  value; };
template <> struct Eps<double> { constexpr static double default_value = 1e-5f; double value; };

#endif  // GRID_TENSOR_PRECISION_H
