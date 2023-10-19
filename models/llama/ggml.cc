//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// This is an outline of the llama.cpp file loader.
//
// Note that llama.cpp uses the term tensors when loading the weights of the model.
//
// All tensor information (or weights) are stored in illama_load_tensor_map structure:
//
// struct llama_load_tensors_map
//  std::vector<llama_load_tensor> tensors;
//  std::unordered_map<std::string, size_t> name_to_idx;
//
// The unordered_mapr maps the name of the tensor to the vector index.
//
// The tensors vector is built in this function:
//
// void read_tensor_metadata(size_t file_idx, llama_load_tensors_map & tensors_map) {
//   while (file.tell() < file.size)
//    llama_load_tensor_shard shard
//     if !find(name)
//       tensors_map.tensors.emplace_back(name);
//     tensors_map.tensors.at(idx).shards.push_back(shard);
//
// It scans the entire file, which is divided into tensor, each of which is
// further divided into 'shards'. A shard represents a column or row of a ...
//
// The simplified routine creates a new tensor in the vector
// if it doesn't exist yet, and adds the shard to the tensor. The shard is initialized with
// the raw data, which is later refined (see below).
//
// The function to read the tensor metadata is called from this constructor for each file:
//
// llama_file_loader()
//  read_magic
//  read_hparams
//  read_vocab
//  read_tensor_metadata
//
// This is called from the llama_model_loader constructor:
//
// llama_model_loader(...)
//   auto * first_file = new llama_file_loader(..., tensor_map)
//   for (...) auto * ith_file = new llama_file_loader(..., tensor_map)
//
// The constructor supports multiple files, and adds all new tensor to the tensor_map, a member
// of the llama_model_loader struct.
//
// The llama_model_loader constructor later walks through all tensors that have been added
// and calls calc_all to refine the details of the tensor and shards.
//
// void calc_all() {
//   calc_type();
//   calc_split_type();
//   calc_ne();
//   calc_size();
//
//
/// ....    void load_data_for(llama_load_tensor & lt) {

#include <string>
#include <iostream>
#include <fstream>

#include <grid/models/llama.h>

#include "ggml.h"

namespace grid {

// FIXME: move to generic headers
template <typename T>
static T MulOvflw(T a, T b)
{
  T ret = a * b;
  if (a != 0 && ret / a != b)
        throw std::runtime_error("multiplication overflow");
  return ret;
}

const std::type_info& GgmlFile::DataType()
{
  return DataType(ftype_);
}


LLaMAFile::FileTensor* GgmlFile::GetFileTensor(const std::string& name, const std::type_info& dtype)
{
  auto lt = tensors_.find(name);
  if (lt == tensors_.end())
    return nullptr;
  // FIXME check dtype
  return &lt->second;
}

// helper function to convert GGML_DATA_TYPE to std::type_info
const std::type_info& GgmlFile::DataType(GgmlDataType dtype)
{
  switch (dtype)
  {
    case kGgmlDataTypeF32:  return typeid(float);
    case kGgmlDataTypeF16:  return typeid(grid::float16_t);
    default:    return typeid(nullptr); // FIXME: what to return?
  }
}


bool GgmlFile::IsSupported(const std::string& path)
{
  std::ifstream file;
  file.open (path, std::ios::in | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("file not opened");

  uint32_t magic;
  file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  if (magic != kGgmlMagicGGJT)
    return false;

  uint32_t version;
  file.read(reinterpret_cast<char *>(&version), sizeof(version));
  if (version != 1)
    return false;

  return true;
}

// Open implements the loading of the weights into ...
// ... it combines the separate steps

// FIXME  map of tensors, each vector can have multiple shards; shard dimentions have to match
LLaMAFile* GgmlFile::Open(const std::string& path, bool mmap)
{
  // FIXME: only mmap supported
  return new GgmlFile(MMapFile(path));
}

// FIXME: why should Load not load directly Tensors, so must be templated?
// FIXME; maybe have a 'check' first, to get type, before actual Load... 
// FIXME: could return type, then do a GgmlFile<TYPE>, so need two versions??
void GgmlFile::Load()
{
  GgmlModelParams ggml_params;
  auto view = mmap_.View();

  view.Read(magic_);
  view.Read(version_);
  view.Read(ggml_params);

  if (ggml_params.ftype_ != kGgmlDataTypeInvalid && ggml_params.ftype_ < kGgmlDataTypeCount)
    ggml_params.ftype_ = GgmlFileToDataType[ggml_params.ftype_];
  ftype_ = ggml_params.ftype_;

  // params_.data_type_ = GetDataType(ggml_params.dtype_); // FIXME: convert to dtype?
  params_.vocab_size_ = ggml_params.vocab_size_;
  params_.hidden_size_ = ggml_params.hidden_size_;
  params_.num_hidden_layers_ = ggml_params.num_hidden_layers_;
  params_.num_attention_heads_ = ggml_params.num_attention_heads_;


  // read vocabs
  vocab_.token_.resize(params_.vocab_size_);
  for (uint32_t i = 0; i < params_.vocab_size_; i++)
  {
    std::string word = view.ReadString();
    float score = 0.0f;
    //if (file_vesion >= 1)
      score = view.Read<float>();
    vocab_.id_[word] = i;
    vocab_.token_[i] = LLaMAVocab::Score{std::move(word), score};
  }

  // read metadata
  while (view.Remaining() > 0)
  {
    // FIXME: move this to separate function? LoadTensor()?
    uint32_t rank = view.Read<uint32_t>();
    if (rank < 1 || rank > kMaxTensorRank)
        throw std::runtime_error("invalid dimension");

    uint32_t namelen = view.Read<uint32_t>();
    GgmlDataType shard_dtype = static_cast<GgmlDataType>(view.Read<uint32_t>());
    // FIXME: file-type differs for quantisized tensors

    uint32_t shape[kMaxTensorRank];
    view.Read(shape, sizeof(shape[0]) * rank);

    std::string name = view.ReadString(namelen);

    auto it = tensors_.find(name);
    if (it != tensors_.end())
    {
      GgmlFileTensor& tensor = it->second;

      // shards need to match
      if (shard_dtype != tensor.dtype_)
        throw std::runtime_error("shard types don't match"); // "inconsistent shard type in '%s'", name.c_str());
      if (rank != tensor.rank_)
        throw std::runtime_error("shard dimensions don't match");
      if (!std::equal(std::begin(tensor.shape_), std::end(tensor.shape_), std::begin(shape)))
          throw std::runtime_error("inconsistens shard shape"); // '%s' in '%s': first shard was '%s'");
    }
    else
    {
      GgmlFileTensor tensor(DataType(shard_dtype));
      bool is_added;

      tensor.rank_ = rank;
      //tensor.dtype_ = shard_dtype;
      std::copy(std::begin(shape), std::end(shape), std::begin(tensor.shape_));

      if (name.find("tok_embeddings") == 0 ||  // 0 here??
          name.find(".attention.wo.weight") != std::string::npos ||
          name.find(".feed_forward.w2.weight") != std::string::npos) // split  by columns
        tensor.transposed_ = true;
      std::tie(it, is_added) = tensors_.emplace(name, tensor);
      // FIXME: why could it fail?
    }

    // if version >= GGJT_V1
    view.Align(32);

    GgmlFileTensor& tensor = it->second;
    TensorShard& shard = tensor.shards_[tensor.shard_count_++];
    shard.address_ = (char*)view.Address() + view.Offset();  // FIXME

    shard.size_ = TypeSize[shard_dtype];
    for (uint32_t i = 0; i < rank; i++)
      shard.size_ = MulOvflw(shard.size_, static_cast<size_t>(shape[i]));
    shard.size_ = shard.size_ / QuantSize[shard_dtype];

    view.SeekCurrent(shard.size_);
  }
}
#if 0
{
  LLaMAFile file = grid::LLaMAFile::Open(path);
  std::unique_ptr<LLamAModel> model;
  if (file.Type() == kDataTypeFp16)
    model = std::make_unique_ptr<grid::LaMAModel>(grid::LLaMAModel<fp16_t>);
  else
    exit(1);

  model->Load();
  std::cout << model->PrintParameters();
  // model->Start("...");
  // signal
  //
  while(true)
    model->Eval("Are electric sheeps dreaming?");
}





} // end of namespace grid    // FIXME


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


#if 0
template<
Tensor* GetTensorArray(const std::string name, const std::vector<uint32_t>& ne)
{
  if (auto it = tensor_map_.idx_.find(name) == tensor_map_.idx_.end()) {
    // error tensor is missing from model
  }

  GgmlFileTensor& t = tensor_map_.tensors_.at(it->second);
  if (lt.ne_ != ne_)
    // error
    ;

  return get_tensor_for(lt, backend);
}
//XX get_tensor_for(FileTensor& lt, /*BACKEND*/)
/*
{
  Tensor* tensor;
  if (lt.ne_.size() == 2)
    Tensor = NewTensor2D(context, lt.type, lt.ne_,at(0), lt.ne_.at(1));
  else
    NewTensor1D(context, lt.type, lt.ne_.at(0));


  set_name(tensor, lt.name_.c_str());

  tensor->backend = backend;
  lt.ggml_tensor = tensor;
  return tensor;
}
*/
#endif
