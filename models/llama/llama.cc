//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <string>
#include <iostream>

#include "llama.h"

#include <grid/tensor/tensor_base.h>
#include <grid/util/demangle.h>


namespace grid {

void LLaMAFile::PrintModelInfo(std::ostream& out) const
{
  auto params = GetParameters();
  // FIXME: some info about the file? out << "File Format ................ Karpathy Snapshot\n";
  out << "Data Type .................. " << grid::Demangle(DataType().name()) << '\n';
  out << "Vocabulary Size ............ " << params.vocab_size_ << '\n';
  out << "Transformer Dimension ...... " << params.embed_size_ << '\n';
  out << "Hidden Dimensions .......... " << params.hidden_dim_ << '\n';
  out << "Number of Layers ........... " << params.num_layers_ << '\n';
  out << "Number of Query Heads ...... " << params.num_heads_ << '\n';
  out << "Number of Key/Value Heads... " << params.num_kv_heads_ << '\n';
  out << "Max Sequence Length ........ " << params.max_seq_len_ << '\n';
}

template <> //template <typename, size_t, auto...> typename Tensor>
LLaMAModel* LLaMAModel::Load<base::Tensor>(const LLaMAFile& file, bool mmap) // FIXME: base::Tensor
{
  if (!mmap)
    throw("only memory-mapped files currently supported");

  auto& data_type =  file.DataType();
  if (data_type == typeid(float))
    return LLaMAModelT<base::Tensor, float>::CreateFrom(file);

  printf("ERR1\n");
  return nullptr;
}


} // end of namespace grid


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
grid::LLaMAFile* grid::LLaMAFile::Open(const std::string& path, Type type)
{
#if 0
  if (grid::GgmlFile::IsSupported(path))
    return grid::GgmlFile::Open(path, mmap);
#endif
  // FIXME: decide on error-handling: exception or return value (nullptr), etc.

  std::cerr << "Unknown file type" << std::endl;
  exit(1);
}

// FIXME: remove and rely on above?
grid::LLaMAModel* grid::LLaMAModel::Open(const std::string& path, bool mmap)
{
  auto* file = grid::LLaMAFile::Open(path, mmap);
  file->Load();

  // FIXME: load the data here?
  if (file->DataType() == typeid(grid::float16_t))
    return new grid::LLaMAModelCPU<grid::float16_t>(file);

  // FIXME: decide on error-handling: exception or return value (nullptr), etc.
  std::cerr << "Unsupported data type" << std::endl;
  exit(1);
}

template <typename Backend>
grid::Model* grid::LLaMAModel::Load(
 {
  //static_assert(Backend != LLaMABackendCPU, "backend not supported");

  if (file->ftype_ == kF16)
    return new LLaMAModelCPU<f16_t>(this);
}

// FIXME: use shared_ptr or unique_ptr for file??
template
grid::Model* grid::LLaMAModelCPU::LLaMAModelCPU(LLaMAFile& file)
{
  // Notes
  //  Tensors/Weights/Arrays are defined with the type, i.e. Tensor<f16_t>
  //  Some might need 'float' precision? maybe TensorDouble<f16_t>?
  //  What about quantisized with structures?
  //
  model.norm_ = file_->GetTensor<Type>("norm.weight", {embeddings_});
}
#template <template <typename, size_t, auto...> typename Tensor>
LLaMAModel* LLaMAModel::Create(const std::type_info& type)
{
  if (type == typeid(float))
    return new LLaMAModelT<Tensor, float>();
  return nullptr;
}

template <template <typename, size_t, auto...> typename Tensor>
LLaMAModel* LLaMAFile::Create(bool mmap)
{
  auto& type = DataType();
  LLaMAModel* model = nullptr;

  if (mmap)
  {
    if (type == typeid(float))
      model = new LLaMAModelT<Tensor, float, grid::kMemoryMapped>();
    else if (type == typeid(int32_t))
      model = new LLaMAModelT<Tensor, int32_t, grid::kMemoryMapped>();
  }
#if 0 // FIXME: needs default constructor
  else
  {
    if (type == typeid(float))
      model = new LLaMAModelT<Tensor, float>();
  }
#endif

  if (model != nullptr)
    model->Load(*this);

  return model;
}
#endif
