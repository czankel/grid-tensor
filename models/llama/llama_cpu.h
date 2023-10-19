//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// Note: this file should not be included directly; it is included by llma.h

#ifndef GRID_LLAMA_LLAMA_CPU_H
#define GRID_LLAMA_LLAMA_CPU_H


namespace grid {

template <typename T>
class LLaMAModelCPU : public grid::LLaMAModelT<T, grid::SlowCPU>
{
 public:
  LLaMAModelCPU(grid::LLaMAFile* file) : grid::LLaMAModelT<T,grid::SlowCPU>(file) {}

  //using Load = LLaMAMOdelT<T,Default>::Load();
 
#if 0
template <typename T>
class LLaMAModeCPUT<T, CPU> : public grid::LLaMAModelT<LLaMAModelT
{
 public:
  // TODO: use std::unique_ptr here? or shared_ptr??
  LLaMAModelT(LLaMAFileT<T>* file) : file_(file) {}
  void Forward();

 private:
#endif
};

#if 0
template <typename T>
void LLaMAModelT<T, CPU>::Forward()
{

}
#endif

} // end of namespace grid

#endif  // GRID_LLAMA_LLAMA_CPU_H
