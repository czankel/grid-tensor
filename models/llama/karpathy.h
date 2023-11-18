//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_LLAMA_KARPATHY_H
#define GRID_LLAMA_KARPATHY_H

#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>

#include <grid/tensor/mmap.h>
#include <grid/tensor/tensor.h>
#include <grid/models/llama.h>

#include "llama.h"

namespace grid {

template <template <typename, size_t, typename...> typename Tensor, typename T>
void KarpathyFile::Load(LLaMAModelT<Tensor, T>& model, const LLaMAFile& file) const
{
  // load the tokenizer
  std::ifstream ifs(tokenizer_path_, std::ios::in | std::ios::binary);
  if (!ifs)
    throw std::runtime_error("failed to open tokenizer file");

  int max_token_length = 0;
  ifs.read(reinterpret_cast<char*>(&max_token_length), sizeof(max_token_length));

  model.max_token_length_ = max_token_length;
  model.vocab_.tokens_.resize(parameters_.vocab_size);
  for (int i = 0; i < parameters_.vocab_size; i++)
  {
    float score = 0.0f;
    ifs.read(reinterpret_cast<char*>(&score), sizeof(score));
    int len = 0;
    ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (len > max_token_length)
      throw std::runtime_error("token length exceeds max token length");

    std::string token(len, '\0');
    ifs.read(&token[0], len);

    model.vocab_.ids_[token] = i;
    model.vocab_.tokens_[i] = LLaMAVocab::Score{token, score};
  }

  int fd = open(path_.c_str(), O_RDONLY);
  if (fd == -1)
    throw std::runtime_error("open failed");

  model.mmap_ = std::make_shared<MMap>(fd, file_size_);
  MMapView view(model.mmap_, sizeof(FileParameters));
  view.Seek(sizeof(parameters_));

  auto& param = parameters_;
  int dim = param.dim;
  //int n_heads = param.n_heads;

// FIXME: move to model

  // temporary buffers
  model.input_ = Tensor(dim, Uninitialized<T>{});
  // map weights
  // Tensor(view, {param.vocab_size, dim});
  // Seek view.Size()??
  // FIXME model.embeddings_ = Tensor(view.Array<T>({param.vocab_size, dim}));
  model.embeddings_ = Tensor(view.Array<T>({param.vocab_size, dim}));

  // Karpathy orders the layer weights by the weights instead of layer
  model.layers_.resize(param.n_layers);
  auto& layers = model.layers_;

  for (int i = 0; i < param.n_layers; i++)
    layers[i].rms_att_weight_ = Tensor(view.Array<T>({dim}));
   // FIXME: array does have something.... 
#if 0
  for (i: param.n_layers)
    model.layers_[i].wq_ = Tensor(view.Array<T>({dim, n_kv_heads * head_size}));
    layer.wq_ = Tensor(view.Array<T>({param.num_attention_heads_,dim}));
    layer.wk_ = Tensor(view.Array<T>({param.num_kv_heads_,dim}));
    layer.wv_ = Tensor(view.Array<T>({param.num_kv_heads_,dim}));

    layer.w0 wo?_ = Tensor(view.Array({param.num_attention_heads_,dim}));
      
    layer.ffn_ = Tensor(view.Array({dim})); // 1D??
      
    layer.w1_ = Tensor(view.Array({param.num_attention_heads_,dim}));
    layer.w2_ = Tensor(view.Array({param.num_attention_heads_,dim}));
    layer.w3_ = Tensor(view.Array({param.num_attention_heads_,dim}));
#endif
#if 0
      //w->wq = ptr;
      //ptr += n_layers * p->dim * (p->n_heads * head_size);

      //w->wk = ptr;
      //ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
      wk_ = Tensor(view.Array({dims});

      //w->wv = ptr;
      //ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
      wv_ = Tensor(view.Array({dims});

      //w->wo = ptr;
      //ptr += n_layers * (p->n_heads * head_size) * p->dim;
      wo_ = Tensor(view.Array({dims});

      //w->rms_ffn_weight = ptr;
      //ptr += n_layers * p->dim;
      ffn_weights_ = Tensor(view.Array({dims});

      //w->w1 = ptr;
      //ptr += n_layers * p->dim * p->hidden_dim;
      w1_ = Tensor(view.Array());

      //w->w2 = ptr;
      //ptr += n_layers * p->hidden_dim * p->dim;
      w2_ = Tensor(view.Array());

      //w->w3 = ptr;
      //ptr += n_layers * p->dim * p->hidden_dim;
      w3_ = Tensor(view.Array());
#endif
#if 0
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
#endif
}

} // end of namespace grid

#endif // GRID_LLAMA_KARPATHY_H
