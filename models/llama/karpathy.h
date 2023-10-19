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


// FIXME: merge into generic Load? could even make it static function? Move to karpathy.h?
template <template <typename, size_t, auto...> typename Tensor, typename T, auto... Args>
void KarpathyFile::Load(LLaMAModelT<Tensor, T, Args...>& model, const LLaMAFile& file) const
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

  printf("MAPPING %ld\n", file_size_);
  model.mmap_ = std::make_shared<MMap>(fd, file_size_);
  MMapView view(model.mmap_, sizeof(FileParameters));
  view.Seek(sizeof(parameters_)); // FIXME: allow extra parameter?


  // dim -- llama: dimension of the input tensor
  // hidden_dim
  // kv_dim = n_dims * n_kv_heads / n_heads
  // kv_mul = n_heads / n_kv_heads
  // head_size = dim / n_heads
  //
  // ggml
  // n_embd - input embeddings
  // n_layer
  // n_ctx
  // n_head
  // n_head_kv
  // n_embd_head
  // n_embd_gqa = n_embd / n_gqa;  n_gqa = n_head/n_head_kv
  // n_tokens = batch.n_tokens

  model.embeddings_ = Tensor(view.Array<T>({parameters_->embedsizevocab_size, dim}));

  // temporary buffers
  model.input_ = Tensor(dim, seq_len, Uninitialized<float>);

#if 0
  model.tok_embeddings_ = Tensor(view.Array<T>({p->vocab_size, dim}));
  auto p = &parameters_;
  int dim = p->dim;

  //n_heads = ;


  printf("DIM %d\n", dim);
  model.tok_embeddings_ = Tensor(view.Array<T>({p->vocab_size, dim}));
  model.norm_ = Tensor(view.Array<T>({p->num_attention_heads_, dim}));
  //model.output_ = ;
#endif
#if 0

  attention_weight_{mmap, ptr,};
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;

    ptr += p->vocab_size * p->dim;
      ///???
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
#endif

#if 0
    // FIXME num-attention-heads??
    for (uint32_t i = 0; i < p->num_hidden_layers_; i++)
    {
      auto& layer = model.layers_[i];

      layer.wq_ = Tensor(view.Array({p->num_attention_heads_,dim}));
      layer.wk_ = Tensor(view.Array({p->num_kv_heads_,dim}));
      layer.wv_ = Tensor(view.Array({p->num_kv_heads_,dim}));

      layer.w0 wo?_ = Tensor(view.Array({p->num_attention_heads_,dim}));
      
      layer.ffn_ = Tensor(view.Array({dim})); // 1D??
      
      layer.w1_ = Tensor(view.Array({p->num_attention_heads_,dim}));
      layer.w2_ = Tensor(view.Array({p->num_attention_heads_,dim}));
      layer.w3_ = Tensor(view.Array({p->num_attention_heads_,dim}));
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
    }
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
