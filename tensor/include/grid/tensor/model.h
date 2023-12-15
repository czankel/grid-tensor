//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// FIXME: not really needed??? move to models, at least..? why in Tensor?

#ifndef GRID_TENSOR_MODEL_H
#define GRID_TENSOR_MODEL_H

#include <iostream>

namespace grid {

class Model
{
 public:
  Model() = default;
  virtual ~Model() = default;
};

} // end of namespace grid

#endif // GRID_TENSOR_MODEL_H
