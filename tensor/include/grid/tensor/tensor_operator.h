//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_OPERATOR_H
#define GRID_TENSOR_TENSOR_OPERATOR_H

namespace grid {

/// TensorOperator is a base class for operators and also a wrapper to call the corresponding operator().
template <typename T, size_t TRank, typename TOperator>
class TensorOperator
{
 public:
  using value_type = T;
  TensorOperator(TOperator& op) : op_(op) {}
  // FIXME: make it virtual and drop TOperator template
  auto operator()() const { return op_(); }
 private:
  TOperator& op_;
};

//template <typename T1, typename T2, typename OP>
//class TensorOperator<



} // end of namespace grid

#endif	// GRID_TENSOR_TENSOR_OPERATOR_H


