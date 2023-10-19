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
  using id = uint32_t;
  using token = std::string;

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
template <template <typename, size_t, auto...> typename Tensor, typename T, auto... Args>
class LLaMAModelT : public LLaMAModel
{
  friend class KarpathyFile;

  using Tensor2D = Tensor<T, 2, Args...>;
  using Tensor3D = Tensor<T, 3, Args...>;

 public:
  LLaMAModelT() = default;
  virtual ~LLaMAModelT() = default;

  /// Load (re-)loads the weights from the provided file.
  void Load(const LLaMAFile& file);

  /// Generate
  virtual void Generate(const std::string& prompt, int steps);

  /// CreateFrom creates the LLaMA model from the provided file.
  //template <template <typename, size_t, auto...> typename, typename, auto...>
  //template<>
  static LLaMAModelT<Tensor, T, Args...>* CreateFrom(const LLaMAFile& file);

 protected:
  void EncodeBPE(const std::string& promplt, std::vector<uint32_t>& token_ids);
  void Forward(const std::vector<uint32_t>& token_ids);

 private:
  std::shared_ptr<MMap> mmap_;

  struct LLaMALayer
  {
    Tensor3D wq_;
    Tensor3D wv_;
    Tensor3D wo_;
    Tensor3D ffn_norm_;
    Tensor3D w1_;
    Tensor3D w2_;
    Tensor3D w3_;
  };

  LLaMAFile*  file_;
  LLaMAVocab  vocab_;
  size_t      max_token_length_; // FIXME move to vocab?

  size_t      num_layers_;


  Tensor2D    input_;
  Tensor2D    embeddings_;
  Tensor2D    norm_;
  //Tensor2D    output_; // is this the same as logits_
  Tensor2D    logits_;

  std::vector<LLaMALayer> layers_;
};


template <template <typename, size_t, auto...> typename Tensor, typename T, auto... Args>
void LLaMAModelT<Tensor,T,Args...>::Load(const grid::LLaMAFile& file)
{
  if (file.DataType() != typeid(T))
    throw ("mismatching type");

  if (dynamic_cast<const KarpathyFile*>(&file) != nullptr)
    dynamic_cast<const KarpathyFile*>(&file)->Load<Tensor, T, Args...>(*this, file);
  else
    throw std::runtime_error("invalid file type");
}

template <template <typename, size_t, auto...> typename Tensor, typename T, auto... Args>
LLaMAModelT<Tensor, T, Args...>* LLaMAModelT<Tensor, T, Args...>::CreateFrom(const LLaMAFile& file)
{
  auto* model = new LLaMAModelT<Tensor, T, Args...>();
  model->Load(file);
  return model;
}


// Byte-Pair Encoding
template <template <typename, size_t, auto...> typename Tensor, typename T, auto... Args>
void LLaMAModelT<Tensor, T, Args...>::EncodeBPE(const std::string& prompt, std::vector<uint32_t>& token_ids)
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


template <template <typename, size_t, auto...> typename Tensor, typename T, auto... Args>
void LLaMAModelT<Tensor, T, Args...>::Forward(const std::vector<uint32_t>& token_ids)
{

  for (auto token: token_ids)
    input_.View({1}, {token, 0}) = embeddings_.View({token}, {0, token_ids[token]});

  std::cout

#if 0
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
    cur = Tensor::RmsNorm(inpl);

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


template <template <typename, size_t, auto...> typename Tensor, typename T, auto... Args>
void LLaMAModelT<Tensor, T, Args...>::Generate(const std::string& prompt, int steps)
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
    return Tensor(View.Array<double>(ft_shards_->address_, {ft->dims_[0]}));
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
