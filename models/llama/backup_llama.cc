//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <string>
#include <iostream>

#include <grid/models/llama.h>

namespace grid {

GgmlLLaMa::~GgmlLLaMa() {}

/// FIXME: rename to LoadGgml?? LLaMa should be same 
ssize_t FileGgml::Load(std::string path)
{
  auto mmap_ = MMapFile(path);
  auto view = mmap_.View();

  uint32_t magic = view.Read<uint32_t>();
  uint32_t version = view.Read<uint32_t>();

  std::cout << "magic: " << magic << " is " << kGgmlMagicGGJT << std::endl;
  std::cout << "version: " << version << std::endl;

  hparams_ = view.Read<HParams>();
  std::cout << "vocab_size: " << hparams_.vocab_size_ << std::endl;

  // read vocabs
  vocab_.token_.resize(hparams_.vocab_size_);
  for (uint32_t i = 0; i < hparams_.vocab_size_; i++)
  {
    std::string word = view.ReadString();
    float score = 0.0f;
    //if (file_vesion >= 1)
      score = view.Read<float>();
    std::cout << "idx " << i << " len " << word.size() << " word " << word <<  " score " << score << std::endl;
    vocab_.id_[word] = i;
    vocab_.token_[i] = Vocab::Score{std::move(word), score};
  }

  // read tensor metadata
  while (view.Remaining() > 0)
  {
    FileTensorShard shard;

    uint32_t dims = view.Read<uint32_t>();
    if (dims < 1 || dims > 2)
      throw();

    uint32_t namelen = view.Read<uint32_t>();

    shard.type = view.Read<uint32_t>();
    // TODO: validate shard.type

    shard.ne.resize(dims);
    view.Read(shard.ne.data(), sizeof(shard.ne[0]) * dims);
    std::string name = view.ReadString(namelen);
    std::cout << "Name " << name << std::endl;

    // if version >= GGJT_V1
    view.Align(32);

    shard.file_off_ = view.Address();

    // calc_size()
    // size = llama_calc_tensor_size(ne, type)
    for (uint32_t dim : ne)
      size = checked_mul<size_t>(size, dim);
    return size / ggml_blck_size(type);

    view.SeekCurrent(shard.size_);

    size_t idx;
    auto it = tensors_map_.idx.find(name);
    if (it != tensors_map_.idx.end())
      idx = it->second;
    else
    {
      idx = tensors_map_.tensors.size();
      tensors_map_.tensors.emplace_back(name);
      tensors_map_.idx.emplace(name, idx);
    }


    /*
    // if  version >= GGJT_V1
    view.Align(sizeof(uint32_t));
    SHARD.file_idx_ = file_idx;
    SHARD.Size();
    */

    // seek to sharedsize??
    //


  }


  return mmap_.Size();
}

void GgmlLLaMa::PrintModelInfo(std::ostream& stream)
{
  stream << "HELLO HERE IS LLAMA" << std::endl;
}


} // end of namespace grid


