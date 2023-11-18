//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_LLAMA_LLAMA_H
#define GRID_LLAMA_LLAMA_H

#include <memory>
#include <unordered_map>
#include <vector>

#include <grid/models/llama.h>
#include <grid/tensor/mmap.h>
#include <grid/tensor/tensor.h>


namespace grid {

/// LLaMAVocab contains the 'vocabs' (vocabularies).
struct LLaMAVocab
{
  // FIXME: ... rename?  token -> symbol, token_id -> token
  // FIXME: token name, token id, token symbol, ...
  //token_id::type = uint32_t
  //token::type = std::string
  using id = uint32_t;
  using token = std::string;

  // FIXME: rename, it's index -> symbol, score
  struct Score
  {
    token text;
    float score;
  };

  // TODO: are both members requiers or can it be combined?
  std::unordered_map<token, id> ids_;
  std::vector<Score> tokens_;
};


/// LLaMAModelT is the templated version of the LLaMAModel class for data type and backend.
///
/// The base implementation .... for a specific data type  FIXME
/// and "backend". Each backend should provide a specific class, for example: LLaMAModelCPU.
template <template <typename, size_t, typename...> typename Tensor, typename T>
class LLaMAModelT : public LLaMAModel
{
  friend class KarpathyFile;

  /// Using two allocators, default for dynamic tensors, and memory-mapped for file tensors.
  using Tensor1D = Tensor<T, 1>;
  using Tensor2D = Tensor<T, 2>;
#if 0
  using TensorFile1D = Tensor<T, 1, MemoryMapped<T>>;
  using TensorFile2D = Tensor<T, 2, MemoryMapped<T>>;
#endif
  using TensorFile1D = Tensor<T, 1, NoAllocator>;
  using TensorFile2D = Tensor<T, 2, NoAllocator>;

 public:
  LLaMAModelT() = default;
  virtual ~LLaMAModelT() = default;

  /// Load (re-)loads the weights from the provided file.
  void Load(const LLaMAFile& file);

  /// Generate
  virtual void Generate(const std::string& prompt, int steps);

  /// CreateFrom creates the LLaMA model from the provided file.
  //template <template <typename, size_t, typename...> typename, typename, typename...>
  //template<>
  static LLaMAModelT<Tensor, T>* CreateFrom(const LLaMAFile& file);

 protected:
  void EncodeBPE(const std::string& promplt, std::vector<uint32_t>& token_ids);
  std::string Decode(uint32_t token_id);  // FIXME: uint32 to token..
  uint32_t Forward(const std::vector<uint32_t>& token_ids); // FIXME: uint32 to token_id::type

 private:
  std::shared_ptr<MMap> mmap_;

  // Keep weights of the layers in different structs instead of using Tensor3D.
  struct LLaMALayer
  {
    // Weights for matmul (note: dim = n_heads * head_size)
    TensorFile2D  wq_;              // {dim, n_heads * head_size}
    TensorFile2D  wk_;              // {dim, n_kv_heads * head_size}
    TensorFile2D  wv_;              // {dim, n_kv_heads * head_size}
    TensorFile2D  wo_;              // {n_heads * head_size, dim}

    //??Tensor2D ffn_norm_;         // {n_heads * head_size, embed_size}
    // Weights for FFN
    TensorFile2D  w1_;               // {hidden_dim, dim}
    TensorFile2D  w2_;              // {dim, hidden_dim}
    TensorFile2D  w3_;              // {hidden_dim, dim}

    TensorFile1D  rms_att_weight_;  // {dim}
    TensorFile1D  rms_ffn_weight_;  // {dim}
    TensorFile1D  rms_final_weight_;// {dim}

    // Local tensors
    Tensor2D      attention_scores_;// {n_heads, max_sequnce_length} // FIXME needed
    Tensor2D      key_cache_;       // {dim, max_sequence_length} ?? FIXME kv_dim??
    Tensor2D      value_cache_;     // {dim, max_sequence_length} ?? FIXME

    Tensor1D      logits_;          // {max_sequence_length}  FIXME: needed?

#if 0
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
#endif
  };

  LLaMAFile*    file_;
  LLaMAVocab    vocab_;
  size_t        max_token_length_; // FIXME move to vocab?

  size_t        num_layers_;

  TensorFile2D  embeddings_;

  // pre-allocated tensors for the forward calculations
  Tensor1D      input_;

  //Tensor2D      norm_;
  //Tensor2D    output_; // is this the same as logits_
  //Tensor2D      logits_;


  std::vector<LLaMALayer> layers_;
};


template <template <typename, size_t, typename...> typename Tensor, typename T>
void LLaMAModelT<Tensor,T>::Load(const grid::LLaMAFile& file)
{
  if (file.DataType() != typeid(T))
    throw ("mismatching type");

  if (dynamic_cast<const KarpathyFile*>(&file) != nullptr)
    dynamic_cast<const KarpathyFile*>(&file)->Load<Tensor, T>(*this, file);
  else
    throw std::runtime_error("invalid file type");

  // initialize ...
  //key_cache_ = Tensor2D(...);
}

template <template <typename, size_t, typename...> typename Tensor, typename T>
LLaMAModelT<Tensor, T>* LLaMAModelT<Tensor, T>::CreateFrom(const LLaMAFile& file)
{
  auto* model = new LLaMAModelT<Tensor, T>();
  model->Load(file);
  return model;
}


// Byte-Pair Encoding
template <template <typename, size_t, typename...> typename Tensor, typename T>
void LLaMAModelT<Tensor, T>::EncodeBPE(const std::string& prompt, std::vector<uint32_t>& token_ids)
{

  token_ids.push_back(kBOS);

  // split text into characters; handle utf-8 characters

  std::string symbol;
  for (size_t i = 0, utf_idx = 0; i < prompt.size(); i++)
  {
    char c = prompt[i];

    symbol.push_back(c);
    if (c < 0 && utf_idx++ < 4)
      continue;

    auto it = vocab_.ids_.find(symbol);
    if (it != vocab_.ids_.end())
      token_ids.push_back(it->second);
    else for (size_t j = 0; j < utf_idx; j++)
      token_ids.push_back(symbol[j] + 3);

    symbol.clear();
    utf_idx = 0;
  }

  while (1)
  {
    float best_score = -1e10; // FIXME: set minimum...
    int   best_token_id = -1;
    int   best_token_idx = -1;

    for (size_t i = 0; i < token_ids.size() - 1; i++)
    {
      auto symbol = vocab_.tokens_[token_ids[i]].text + vocab_.tokens_[token_ids[i + 1]].text;
      auto it = vocab_.ids_.find(symbol);

      float score;
      if (it != vocab_.ids_.end() && (score = vocab_.tokens_[it->second].score) > best_score)
      {
        best_score = score;
        best_token_id = it->second;
        best_token_idx = i;
      }
    }

    if (best_token_idx == -1)
      break;

    token_ids[best_token_idx] = best_token_id;
    token_ids.erase(token_ids.begin() + best_token_idx + 1);
  }

  token_ids.push_back(kEOS);

  if (token_ids.size() == 2)
    throw std::runtime_error("expected at least 1 prompt token");
}


template <template <typename, size_t, typename...> typename Tensor, typename T>
std::string LLaMAModelT<Tensor, T>::Decode(uint32_t token_id)
{
  std::string symbol = vocab_.tokens_[token_id].text;
  // FIXME if first token after <BOS> drop space, add and use prev_token_id?
  // FIXME: convert raw bytes <0x01> to actual bytes

  return symbol;
}


// Note that this is a "lower-rank" implementation going through the calculation for each
// token vector instead of combining a sequence into a matrix and using higher-rank tensors.
template <template <typename, size_t, typename...> typename Tensor, typename T>
uint32_t LLaMAModelT<Tensor, T>::Forward(const std::vector<uint32_t>& token_ids) // FIXME: uint32_t to token_id::type
{
  // FIXME: need a last position??
  //size_t pos = 0;
  //for (auto token: token_ids)
    //input_.View({1}, {0, pos++}) = embeddings_.View({1}, {0, token_ids[token]});

  for (auto token: token_ids)
  {
    //input_ = embeddings_.View({1}, {0, token_ids[token]});
  }
#if 0
    for (auto l: layers_)
    {
      using Broadcast = grid::Broadcast;
      // vec o vec ('hardamard' multiplication by broadcasting both vectors to (N, 1))
      l.x_ = grid::TensorRmsNorm(input_.View({0, Broadcast}) * l.rms_att_weight_.View({0, Broadcast}));
      l.x_ = grid::TensorRmsNorm(input_.View({0,-1}) * l.rms_att_weight_.View({0,-1}));

      // add (note that key is transposed)
      l.q_.View({0},{pos, 0}) = x_ * l.wq_;                  // vec(dim) x mat(dim,dim) -> vec(dim)
      l.key_cache_.View({0},{pos, 0}) = x_ * l.wk_;          // mat x vec?? FIXME vec(dim) x mat(dim,kv_dim) -> vec(kv_dim)
      l.val_cache_.View({1},{pos, 0}) = x_ * l.wv_;          // vec(dim) x mat(dim,kv_dim) -> vec(kv_dim)

      // RoPE, rotate each 'head'
      auto q = q_.Data();
      auto k = key_cache_.Data();

      for (int i = 0; i < dim_; i+=2)
      {
        float rot = (float) pos / powf(10000.0f, (float)(i % head_size) / (float)head_size);
        float fcr = cosf(rot);
        float fci = sinf(rot);

        float v0 = q[i];
        float v1 = q[i+1];
        q[i]   = v0 * fcr - v1 * fci;
        q[i+1] = v0 * fci + v1 * fcr;

        if (i < kv_dim_)
        {
          float v0 = k[i];
          float v1 = k[i+1];
          k[i]   = v0 * fcr - v1 * fci;
          k[i+1] = v0 * fci + v1 * fcr;
        }
      }
#endif

#if 0
      // multihead attention
      // mat(pos+1,dim) x vec(dim) -> vec(pos+1)
      l.attention_ = l.q_ * l.key_cache_.View({0,1},strides,{,pos}) / sqrtf(head_size); // FIXME: sqrtf const?
      l.attention_ = TensorSoftMax(l.attention_);

      l.x2_ = l.value_cache_ * l.attention_; // mat(dim,..) x vec() -> vec(...);
      l.x_ += l.x2_ * l.wo_;  // vec(dim) x mat(dim,dim) -> vec(dim) // FIXME mat x vec?
                              //
      // FFN RMS Norm
      xxx = TensorRmsNorm(l.x_.View({0, grid::Broadcast}) * l.rms_ffn_weight_.View({0, grid::Broadcast});

      // FFN: (self.w2(F.silu(self.w1(x)) * self.w3(x))
      x_ += TensorSwiGLU(xb_ * w1 * w2) * w3_; // mat(dim, hidden_dim) -> ...
    }
    TensorRmsNorm(x_, rms_final_weight_);   // vec(dim), vec(dim) -> vec(dim);
    logit = x_ * wcls_; // vec(dim) x vec(dim) -> scalar
    // FIXME: wcls is optionaL?
    logits_.append(TensorRmsNorm(x_ * rms_final_weight_) * wcls);
  }
#endif
  return 0; // FIXME logits_;

#if 0
  // classifier
  logits_ = x_ * rms_final_weight * wcls_;
  Tensor<int32_t> embedded({ntokens}, tokens)
  //memcpy(embd->data, tokens, ntokens x elem_size(embd));
  Tensor input = TensorProduct(tok_embeddings_, tokens);
  std::cout << input;

  // inpl = get_rows(model.tok_embeddings, embd); --> ROWS, which rows (plural)?
  input = token_embeddings_.Row(seq);
  input = token_embeddings_.View({1}, {0, seq});

  for (int il = 0; il < nlayers; il++)
  {
    Tensor inpSA = inpl;
    cur = Tensor::RmsNorm(inpl);  // up to rank-3

    // cur = cur*attention_norm(broadcasted)
    cur = mul(cur, model.layers[il].attention_norm);  // Rank X x Rank 2   ??

    // self-attention
    // compute q and k and RoPE them
    Tensor tmpk = Tensor::MatMul(model.layers[il].wk, cur);
    Tensor tmpq = Tensor::MatMul(mdoel.layers[il].wq, cur);

    Tensor kCur = rope_inplace(reshape_3d(tmpk, nembedding/nheads, nheads, N), n_past, n_rot, 0, 0);
    Tensor Qcur = rope_inplace(reshape_3d(tmpq, nembd/nhead, nhead, N), n_post, n_rot, 0, 0);

    // store key and value to memory
    // compute the transposed [N, n_embd] V matrix
    Tensor tmpv = MatMul(model.layers[i].wv, cur);
    Tensor Vcur = transpose(reshape_2d(tmpv, n_mebd, N));

    Tensor k = view_1d(kv_self.k, N * n_mebd, (....));
    Tensor v = view_2d(kv_self.v, N, n_embd, ...);

    // important, storing RoPEed version of K in the KV cache!
    build_forward_expand(&gf, cpy(Kcur, k));
    build_forward_expand(&gf, cpy(Vcur, v));

    Tensor Q = permute(Qcur, 0, 2, 1, 3);
    Tensor K = permute(reshape_3d(view_1d(kv_self.k, (n_past + N)*n_embd, il*n_ctx*element_size...)));

    // K + Q
    Tensor KQ = MatMul(K, Q);
    Tensor KQ_scale = new_f32(1.0f/sqrtf(float(n_embd)/n_head));
    Tensor KQ_scaled = scale_inplace(KQ, KQ_scale);
    Tensor KQ_masked = diag_mask_inf_inplace(KQ_scaled, n_post);
    Tensor KQ_soft_max = soft_max_inplace(KQ_masked);
  Tensor& V = view_3d(kv_self.v, n_past + N, n_embd/n_head, n_head, n_ctx...);

  Tensor& KQV = MatMul(V, KQ_soft_max);

  Tensor& KQV_merged = permute(KQV, 0, 2, 1, 3);

  cur = cpy(KQV_merged, new_tensor_2d(f32_t, n_mebd, N));
  cur = MatMul(model.layers[il].wo, cur);
  }
#endif

}


template <template <typename, size_t, typename...> typename Tensor, typename T>
void LLaMAModelT<Tensor, T>::Generate(const std::string& prompt, int steps)
{
  std::vector<uint32_t> token_ids;
  EncodeBPE(prompt, token_ids);

  // FIXME: remove
  for (int i: token_ids)
    std::cout << vocab_.tokens_[i].text << '\n';

  // Generate

  //for (int pos = 0; pos < steps; pos++)
  Forward(token_ids);



    /*
    if (pos < num_tokens - 1)
      next = tokens[pos + 1];
    else
      next = sample();

    if (next == kEOS)
      break;

    xx = Decode(token, next);
    out << xx;  // FIXME flush?
    */
}

} // end of namespace grid

// include supported file formats

#include "karpathy.h" // FIXME: mihgt need to move to end, or split this file

#endif  // GRID_LLAMA_LLAMA_H

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
#if 0
#if 0
  // FIXME: only allows to get tensor of model type; can expand with template <typename U>
  Tensor GetTensor1D(const std::string name)
  {
    LLaMAFile::FileTensor* ft = file_->GetFileTensor(name, typeid(T));
    return Tensor(View.Array<float>(ft_shards_->address_, {ft->dims_[0]}));
    // ft->dims_[0], ft->shards_->address_, mmap_);
  }

  Tensor<T, _Device> GetTensor2D(const std::string name)
  {
    LLaMAFile::FileTensor* ft = file_->GetFileTensor(name, typeid(T));
    //return Tensor<T, _Device>(ft->rank_, ft->dims_, ft->shard_count_, ft->shards_);
    return Tensor<T, _Device>(ft->dims_[0], ft->dims_[1], static_cast<T*>(ft->shards_->address_));
  }
#endif

// FIXME: part of constructor or initialize?
template <typename T, typename Backend>
void LLaMAModelT<T,Backend>::Load()
GgmlFile::Load(Model)
{
  LLaMAModel::Parameters& params = file_->GetParameters();

  tok_embeddings_ = GetTensor("tok_embeddings.weight", 2);
  norm_           = GetTensor("norm.weight", 2);
  output_         = GetTensor("output.weight", 2);

  layers_.resize(params.num_hidden_layers_);
  for (uint32_t i = 0; i < params.num_hidden_layers_; i++)
  {
    auto& layer = layers_[i];
    const std::string layer_name = "layers." + std::to_string(i);

    layer.attn_norm_ = GetTensor2D(layer_name + ".attention_norm_weight");
    layer.wq_        = GetTensor2D(layer_name + ".attention.wq.weight");
    layer.wk_        = GetTensor2D(layer_name + ".attention.wk.weight");
    layer.wv_        = GetTensor2D(layer_name + ".attention.wv.weight");
    layer.wo_        = GetTensor2D(layer_name + ".attention.wo.weight");
    layer.ffn_norm_  = GetTensor2D(layer_name + ".ffn_norm.weight");
    layer.w1_        = GetTensor2D(layer_name + ".feed_forward.w1.weight");
    layer.w2_        = GetTensor2D(layer_name + ".feed_forward.w2.weight");
    layer.w3_        = GetTensor2D(layer_name + ".feed_forward.w3.weight");
  }
}





};
#endif
