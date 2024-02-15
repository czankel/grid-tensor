//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <string>
#include <iostream>

#include <grid/tensor/tensor_base.h>
#include <grid/util/demangle.h>

#include "llama.h"
#include "karpathy.h"
#include "ggml.h"

namespace grid {

//
// LLaMAFile
//

LLaMAFile* LLaMAFile::Open(Type file_type, std::string_view model_path)
{
  LLaMAFile* file;

  switch (file_type)
  {
    case kKarpathy: file = new grid::KarpathyFile(model_path); break;
    case kGgml:     file = new grid::GgmlFile(model_path); break;
    default: throw std::runtime_error("invalid model file type: " + std::to_string(file_type));
  }

  file->Load();
  return file;
}


LLaMAFile* LLaMAFile::Open(Type file_type, std::string_view model_path, std::string_view tokenizer_path)
{
  LLaMAFile* file;

  switch (file_type)
  {
    case kKarpathy: file = new grid::KarpathyFile(model_path, tokenizer_path); break;
    default: throw std::runtime_error("file already includes tokenizer or is invalid");
  }

  file->Load();
  return file;
}


std::ostream& LLaMAFile::PrintModelInfo(std::ostream& out) const
{
  LLaMAModel::Parameters params;
  GetParameters(params);
  out << "Data Type .................. " << grid::Demangle(DataType().name()) << '\n';
  out << "Vocabulary Size ............ " << params.vocab_size_ << '\n';
  out << "Transformer Dimension ...... " << params.dim_ << '\n';
  out << "Hidden Dimensions .......... " << params.hidden_dim_ << '\n';
  out << "Number of Layers ........... " << params.num_layers_ << '\n';
  out << "Number of Query Heads ...... " << params.num_heads_ << '\n';
  out << "Number of Key/Value Heads... " << params.num_kv_heads_ << '\n';
  out << "Max Sequence Length ........ " << params.max_seq_len_ << '\n';

  return out;
}

//
// LLaMAModel
//

LLaMAModel* LLaMAModel::Load(LLaMAFile& file, bool mmap)
{
  if (!mmap)
    throw("only memory-mapped files currently supported");

  // TODO: because the model is templated, all supported data types need to be specialized here.
  auto& data_type =  file.DataType();
  if (data_type == typeid(float))
    return LLaMAModelT<float>::Load(file);
  else if (data_type == TypeID<float16_t>())
    return LLaMAModelT<float16_t>::Load(file);

  throw std::runtime_error("invalid data type, only float is supported");

}

} // end of namespace grid
