//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <fstream>

#include <grid/models/llama.h>


namespace grid {

// FIXME: is the "Open" or more like "Info"?
KarpathyFile::KarpathyFile(const std::string& path, const std::string& tokenizer_path)
  : tokenizer_path_(tokenizer_path),
    path_(path)
{
  // FIXME: out or in?
  std::ifstream ifs(path, std::ios::out | std::ios::binary);
  if (!ifs)
    throw std::runtime_error("failed to open file");

  ifs.read(reinterpret_cast<char*>(&parameters_), sizeof(parameters_));

  ifs.seekg(0, ifs.end);
  file_size_ = ifs.tellg();
  ifs.close();

  if (!ifs.good())
    throw std::runtime_error("failed to read file");
}


// FIXME what's supported? make this constexr? inline?static?
const std::type_info& KarpathyFile::DataType() const
{
  return typeid(float);
}


LLaMAModel::Parameters KarpathyFile::GetParameters() const
{
  return LLaMAModel::Parameters {
    parameters_.vocab_size,
    parameters_.dim,
    parameters_.hidden_dim,
    parameters_.n_layers,
    parameters_.n_heads,
    parameters_.n_kv_heads,
    parameters_.max_seq_len
  };
}

#if 0
    int dim; // transformer dimension (embed_size)
    int hidden_dim; // for ffn layers (??)
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length


#endif
#if 0
template <template <typename, size_t, auto...> typename Tensor, typename T, auto... Args>
KarpathyModelT::Load(LLaMAModelT<Tensor, T, Args...>& model)
{
#if 0
  void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;

////

  // rms weight: layers * dim
  // wq          n_layers * dim * n_heads * head_size
  // wk          n_layers * dim * n_kv_heads * head_size
  // wv          n_layers * dim * n_kv_heads * head_size
  // swo         n_layers * n_heads * head_size * dim
  // ffn         n_layers * dim
  // w1          n_layers * dim * hidden_dim
  // w2          n_layers * hidden_dim * dim
  // w3          n_layers * dim * hidden_dim
  // rms final   dim
  // <skip>      seq_len * head_size / 2 (freq_cis_real for RoPE)

    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
#endif
}
#endif

} // end of namespace grid
