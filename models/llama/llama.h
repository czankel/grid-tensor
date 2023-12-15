//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef _LLAMA_H
#define _LLAMA_H

#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include <grid/models/llama.h>

#include <grid/tensor/mmap.h>
#include <grid/tensor/tensor.h>

#include "llama_vocab.h"

namespace grid {

/// LLaMAModelT is the templated version of the LLaMAModel class for data type and backend.
///
/// The base implementation .... for a specific data type  FIXME
/// and "backend". Each backend should provide a specific class, for example: LLaMAModelCPU.
template <template <typename, size_t, typename...> typename Tensor, typename T>
class LLaMAModelT : public LLaMAModel
{
  friend class KarpathyFile;
  friend class GgmlFile;

  /// Using two allocators, default for dynamic tensors, and memory-mapped for file tensors.
  using Tensor1D = Tensor<T, 1>;
  using Tensor2D = Tensor<T, 2>;
  using TensorFile1D = Tensor<T, 1, NoAllocator>;
  using TensorFile2D = Tensor<T, 2, NoAllocator>;

 protected:
  LLaMAModelT() = default;

 public:
  virtual ~LLaMAModelT() = default;

  // LLaMAModel::
  virtual void Predict(std::string_view prompt, size_t steps);
// FIXME: pass const file, but file must be initialize from file data, such as parameters... do that in Open?
  /// Load loads the LLaMA model from the provided file.
  static LLaMAModelT<Tensor, T>* Load(LLaMAFile& file);

 protected:
  // EncodeBPE encodes the prompt into a token vector using byte-pair encoding
  void EncodeBPE(std::string_view prompt, std::vector<uint32_t>& token_ids);

  // Decode decodes the provided current token with the FIXME
  std::string Decode(LLaMAVocab::token , LLaMAVocab::token);

  /// Forward runs a single forward run through themodel for the provided token and FIXME size?
  void Forward(LLaMAVocab::token token, size_t);

  /// Sample samples the current logits .. FIXME
  LLaMAVocab::token Sample();
  LLaMAVocab::token SampleArgMax();

 private:
  LLaMAModel::Parameters parameters_;
  std::shared_ptr<MMap>  mmap_;
  LLaMAVocab             vocab_;
  size_t                 max_token_length_;

  struct LLaMALayer
  {
    // (note that dim = n_heads * head_size and n_kv_heads = n_heads for this implementation)
    TensorFile2D  wq_;              // {dim, n_heads * head_size}
    TensorFile2D  wk_;              // {dim, n_kv_heads * head_size}
    TensorFile2D  wv_;              // {dim, n_kv_heads * head_size}
    TensorFile2D  wo_;              // {n_heads * head_size, dim}

    // Weights for FFN
    TensorFile2D  w1_;              // {hidden_dim, dim}
    TensorFile2D  w2_;              // {dim, hidden_dim}
    TensorFile2D  w3_;              // {hidden_dim, dim}

    TensorFile1D  att_norm_;         // {dim}
    TensorFile1D  ffn_norm_;         // {dim}

    // Runtime tensors
    Tensor2D      key_cache_;       // {kv_dim, max_sequence_length}
    Tensor2D      value_cache_;     // {kv_dim, max_sequence_length}
    Tensor1D      q_;
  };

  TensorFile2D  embeddings_;
  TensorFile1D  output_norm_;        // {dim}
  TensorFile2D  output_;            // {vocab_size, dim}

  // Runtime tensors
  Tensor1D      x_;                 // {dim}
  Tensor1D      xb_;
  Tensor1D      logits_;            // output {vocab_size}
  Tensor1D      scores_;            // {n_heads * head_size}

  std::vector<LLaMALayer> layers_;
};


template <template <typename, size_t, typename...> typename Tensor, typename T>
LLaMAModelT<Tensor, T>* LLaMAModelT<Tensor, T>::Load(LLaMAFile& file)
{
  auto* model = new LLaMAModelT<Tensor, T>();

  file.GetParameters(model->parameters_);
  file.GetTokenizer(model->vocab_);

  model->mmap_ = std::make_shared<MMap>(file.MapTensors());
  char *base = static_cast<char*>(model->mmap_->Address());

  auto& params = model->parameters_;
  size_t n_layers =   params.num_layers_;
  size_t hidden_dim = params.hidden_dim_;
  size_t dim =        params.dim_;
  size_t kv_dim =     dim * params.num_kv_heads_ / params.num_heads_;

  model->layers_.resize(n_layers);
  for (size_t i = 0; i < n_layers; i++)
  {
    auto& layer = model->layers_[i];
    layer.att_norm_ = Tensor({dim}, file.GetTensor<T>(base, LLaMAFile::kAttentionNorm, i));
    layer.wq_ =       Tensor({dim, dim}, file.GetTensor<T>(base, LLaMAFile::kAttentionQuery, i));
    layer.wk_ =       Tensor({kv_dim, dim}, file.GetTensor<T>(base, LLaMAFile::kAttentionKey, i));
    layer.wv_ =       Tensor({kv_dim, dim}, file.GetTensor<T>(base, LLaMAFile::kAttentionValue, i));
    layer.wo_ =       Tensor({dim, dim}, file.GetTensor<T>(base, LLaMAFile::kFeedForwardWo, i));
    layer.ffn_norm_ = Tensor({dim}, file.GetTensor<T>(base, LLaMAFile::kFeedForwardNorm, i));
    layer.w1_ =       Tensor({hidden_dim, dim}, file.GetTensor<T>(base, LLaMAFile::kFeedForwardW1, i));
    layer.w2_ =       Tensor({dim, hidden_dim}, file.GetTensor<T>(base, LLaMAFile::kFeedForwardW2, i));
    layer.w3_ =       Tensor({hidden_dim, dim}, file.GetTensor<T>(base, LLaMAFile::kFeedForwardW3, i));
  }

  model->embeddings_ =  Tensor({params.vocab_size_, dim}, file.GetTensor<T>(base, LLaMAFile::kEmbeddings));
  model->output_norm_ = Tensor({dim}, file.GetTensor<T>(base, LLaMAFile::kOutputNorm));
  model->output_ =      Tensor({params.vocab_size_, dim}, file.GetTensor<T>(base, LLaMAFile::kOutput));

  // Initialize runtime tensors
  model->x_ =       Tensor(dim, Uninitialized<T>{});
  model->xb_ =      Tensor(dim, Uninitialized<T>{});
  model->logits_ =  Tensor(params.vocab_size_, Uninitialized<T>{});
  model->scores_ =  Tensor(dim, Uninitialized<T>{});

  for (size_t i = 0; i < n_layers; i++)
  {
    auto& layer = model->layers_[i];
    layer.key_cache_ =   Tensor({params.max_seq_len_, kv_dim}, T(0));
    layer.value_cache_ = Tensor({params.max_seq_len_, kv_dim}, T(0));
    layer.q_ =           Tensor(dim, Uninitialized<T>{});
  }

  return model;
}


// Byte-Pair Encoding
template <template <typename, size_t, typename...> typename Tensor, typename T>
void LLaMAModelT<Tensor, T>::EncodeBPE(std::string_view prompt, std::vector<LLaMAVocab::token>& tokens)
{
  if (vocab_.add_bos_token_)
    tokens.push_back(vocab_.bos_token_);

  // FIXME
  std::string underscore = "▁";
  auto s = vocab_.tokens_.find(underscore);
  if (s == vocab_.tokens_.end())
    printf("NO UNDERSCORE!\n");

  // split text into characters; handle utf-8 characters
  std::string symbol;
  for (size_t i = 0, utf_idx = 0; i < prompt.size(); i++)
  {
    char c = prompt[i];
    if (c  == ' ')
    {
      tokens.push_back(s->second);
      continue;
    }

    symbol.push_back(c);
    if (c < 0 && utf_idx++ < 4)
      continue;

    auto it = vocab_.tokens_.find(symbol);
    if (it != vocab_.tokens_.end())
      tokens.push_back(it->second);
    else for (size_t j = 0; j < utf_idx; j++)
      tokens.push_back(symbol[j] + 3);

    symbol.clear();
    utf_idx = 0;
  }

  while (1)
  {
    LLaMAVocab::token best_token;
    // FIXME: is vocab always float?
    float best_score = std::numeric_limits<float>::lowest();
    int   best_index = -1;

    std::cout << "TOKENS: ";
    for (size_t i = 0; i < tokens.size(); i++)
      std::cout << tokens[i] << ' ';
    std::cout << '\n';

    for (size_t i = 0; i < tokens.size() - 1; i++)
    {
      auto symbol = vocab_.scores_[tokens[i]].text + vocab_.scores_[tokens[i + 1]].text;
      auto it = vocab_.tokens_.find(symbol);

      float score;
      if (it != vocab_.tokens_.end() && (score = vocab_.scores_[it->second].score) > best_score)
      {
        best_score = score;
        best_token = it->second;
        best_index = i;
      }
    }

    if (best_index == -1)
      break;

    tokens[best_index] = best_token;
    tokens.erase(tokens.begin() + best_index + 1);
  }

  if (vocab_.add_eos_token_)
    tokens.push_back(vocab_.eos_token_);

  if (tokens.size() == 2)
    throw std::runtime_error("expected at least 1 prompt token");
}


template <template <typename, size_t, typename...> typename Tensor, typename T>
std::string LLaMAModelT<Tensor, T>::Decode(LLaMAVocab::token prev, LLaMAVocab::token token)
{
  std::string symbol = vocab_.scores_[token].text;

  // if first token after <BOS> drop any space
  if (prev == kBOS && symbol[0] == ' ')
    symbol.erase(0,1);

  // FIXME: convert raw bytes <0x01> to actual bytes

  return symbol;
}

// Note that this is a "lower-rank" implementation going through the calculation for each
// token vector instead of combining a sequence into a matrix and using higher-rank tensors.
template <template <typename, size_t, typename...> typename Tensor, typename T>
void LLaMAModelT<Tensor, T>::Forward(LLaMAVocab::token token, size_t pos)
{
  // FIXME unused, remove: constexpr auto BC = grid::Broadcast;
  using namespace grid;

  size_t dim = parameters_.dim_;
  size_t n_heads = parameters_.num_heads_;
  size_t n_kv_heads = parameters_.num_kv_heads_;
  size_t head_size = dim / n_heads;
  size_t kv_dim = parameters_.num_kv_heads_ * head_size;

  x_ = embeddings_.Rearrange({1}, {token, 0});

  for (auto& l: layers_)
  {
    // normalize input and element-multiply with weight.
    xb_ = TensorElemMul(TensorRmsNorm(x_), l.att_norm_);         // [dim] o [dim] -> [dim]

    // Insert Weight(xb) vectors into the key and value caches at row "pos"
    l.key_cache_.Rearrange({1}, {pos, 0}) =   l.wk_ * xb_;      // [kv_dim, dim] * [dim] -> [kv_dim]
    l.value_cache_.Rearrange({1}, {pos, 0}) = l.wv_ * xb_;      // [kv_dim, dim] * [dim] -> [kv_dim]
    l.q_ = l.wq_ * xb_;                                         // [dim, dim] * [dim] -> [dim]

    // RoPE, rotate for each 'head'
    auto q = l.q_.Data();
    auto k = l.key_cache_.Rearrange({1}, {pos, 0}).Data();

    for (size_t i = 0; i < dim; i+=2)
    {
      float rot = (float) pos / powf(10000.0f, (float)(i % head_size) / (float)head_size);
      float fcr = cosf(rot);
      float fci = sinf(rot);

      float v0 = q[i];
      float v1 = q[i+1];
      q[i]   = v0 * fcr - v1 * fci;
      q[i+1] = v0 * fci + v1 * fcr;

      if (i < kv_dim)
      {
        float v0 = k[i];
        float v1 = k[i+1];
        k[i]   = v0 * fcr - v1 * fci;
        k[i+1] = v0 * fci + v1 * fcr;
      }
    }

    // MultiHead(Q,K,V) = concat(head_1, ..., head_h) W_0, with head = Attention(Q_head,K_head,V_head)
    for (size_t head = 0; head < n_heads; head++)
    {
      // FIXME optimize
      size_t head_offset = head * head_size;
      size_t kv_head_offset = (head / (n_heads/n_kv_heads)) * head_size;

      // Attention(Q,K,V) = softmax(Q * K^T / sqrt(kv_dim)) * V
      //
      // For a single token (seq = 1) atpos, and looping through the head count, this
      // reduces to: (Note that K is multiplied with the q vector on the right side, so not transposed)
      // scores {head_size @ head * head_size] =
      //   softmax(K [pos+1, head_size] @ [0,head] * q [head_size] @ [head]) * V [pos+1, head_size] @ [0, head]
      //   scores = softmax(K * q) * V    [pos+1, head_size] * q [head_size] -> [pos+1]
      scores_.Span({head_size}, {head_offset}) =
        TensorSoftMax(l.key_cache_.Span({pos + 1, head_size}, {0, kv_head_offset}) *  // [pos+1, head_size] *
                      l.q_.Span({head_size}, {head_offset}) *                         // [head_size] -> [pos+1]
                      (1.0f / sqrt(head_size))) *                                     // *
        l.value_cache_.Span({pos+1, head_size}, {0, kv_head_offset});                 // [pos+1,head_size] -> [head_size]
    }

    // xb_ = l.wo_ * scores_;

    // bring it all together
    // [dim, dim] * [dim = n_heads * head_size] -> [dim]
    x_ += l.wo_ * scores_;

    // ffn rms norm -> [dim]
    xb_ = TensorElemMul(TensorRmsNorm(x_), l.ffn_norm_);

    // self.w2(F.silu(self.w1(x)) * self.w3(x))  (elem. multiplication)
    // w1(x), w3(x)         -> [hidden_dim, dim] * [dim] = [hidden_dim]
    // silu(w1(x)) o w3(x)  -> [hidden_dim] o [hiddem_dim] = [hidden_dim]
    // w2(...)              -> [dim,hidden_dim] * [hidden_dim] = [dim]
    x_ += l.w2_ * TensorElemMul(TensorSilu(l.w1_ * xb_), l.w3_ * xb_);
  }
  // Final RMS norm and classified into logits
  // [vocab_size,dim] * (Norm([dim] o [dim]) 
  logits_ = output_ * TensorElemMul(TensorRmsNorm(x_), output_norm_); // -> [vocab_size]
}

#if 0
using LLaMA_SAMPLE = tuple<int, float>;
struct sampler
{
  vocab_size_;
  temperature_;
  topp_;
  rng_state_;
  ProbIndex prob_index_;
}
#endif


template <template <typename, size_t, typename...> typename Tensor, typename T>
LLaMAVocab::token LLaMAModelT<Tensor, T>::SampleArgMax()
{
  float max_p = std::numeric_limits<float>::lowest();
  int max_i = 0;
  auto data = logits_.Data();

  for (LLaMAVocab::token i = 0; i < logits_.Dimensions()[0]; i++)
  {
    if (data[i] > max_p)
    {
      max_p = data[i];
      max_i = i;
    }
  }

  return max_i;
}

template <template <typename, size_t, typename...> typename Tensor, typename T>
LLaMAVocab::token LLaMAModelT<Tensor, T>::Sample()
{
  // greedy argmax sampling: return the token with the highest probability
  // if (temperature_ == value_type(0))
    return SampleArgMax();
#if 0
  // apply the temperature to the logits and softmax to get probabilities for the next token
  logits[i] *= 1.0f / temperature_;
  logits = TensorSoftMax(logits);

  // predicted probabiliy distribution or top-p (nucleus) sampling, clamping the least likely tokens to zero
  float entropy = random<float>(&rng_state_);
  if (topp_ <= 0 || topp_ >= 1)
    next = SampleMult(logits, vocab_size_, entropy);
  else
    next = SampleTopp(logits, voab_size_, topp_, probindex, entropy);
#endif
}

#if 0
SampleMult(...)
{
  float cdf = 0.0f;
  for (int i = 0; i < n; i++)
  {
    cdf += probabilities[i];
    if (coin < cdf)
      return i;
  }
  return n-1;
}

SampleTopp(float topp)
{
  std::vector<std::pair<int, float>> sort;

  const float cutoff = (1.0f - topp) / (n - 1);
  int index = 0;
  for (auto val : logits)
    if (val >= cutoff)
      sort.emplace_back(std::make_pair(index++, val));
  std::sort(x.begin(), x.end(), [](float a, float b) { .second > .second });

  float cumulative = 0;
  for (sort::reverse_iterator it = sort.rbegin(); it != sort.rend() && cumulative <= topp; ++it)
    cumulative += it->second;
  sort.erase(it, sort.rbegin())

  float r = entropy * cumulative_porb;
  float cdf = 0;
  for (auto e : sort)
  {
    cdf += e.second;
    if (cdf > r)
      return e.first;
  }

  return sort.end()->first;
}
#endif


template <template <typename, size_t, typename...> typename Tensor, typename T>
void LLaMAModelT<Tensor, T>::Predict(std::string_view prompt, size_t steps)
{
  using token = LLaMAVocab::token;

  std::vector<token> prompt_tokens;
  EncodeBPE(prompt, prompt_tokens);

  size_t pos = 0;
  size_t prompt_token_size = prompt_tokens.size();

  for (token curr = prompt_tokens[0]; pos < steps; pos++)
  {
    Forward(curr, pos);
    token prev = curr;

    curr = (pos < prompt_token_size - 1) ? prompt_tokens[pos + 1] : Sample();
    if (curr == kBOS)
      break;

    std::string symbol = Decode(prev, curr);
    std::cout << symbol << std::flush;
  }
  std::cout << std::endl;
}

} // end of namespace grid

#endif  // GRID_MODEL_LLAMA_H

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
#if 0

// SETUP RNG...
::Init()
{
  rng_seed_ = (unsigned int)time(NULL);
  temperature_ = value_type(0);
  topp = 0.9;
  steps = 0;
}

/*
  if (dynamic_cast<const KarpathyFile*>(&file) != nullptr)
    dynamic_cast<const KarpathyFile*>(&file)->Load<Tensor, T>(*this, file);
  else if (dynamic_cast<const GgmlFile*>(&file) != nullptr)
    dynamic_cast<const GgmlFile*>(&file)->Load<Tensor, T>(*this, file);
  else
    throw std::runtime_error("invalid file type");
*/

  parameters_ = file.GetParameters();
}

SOME_FUNC()
{
  LoadVocab();
  LoadTensors();
  embeddings_ = GetTensor("model_dimensions", {param.vocab_size_, dim}, offset);

  // Pre runtime tensors
  int n_layers = param.n_layers;
  model.layers_.resize(n_layers);
  auto& layers = model.layers_;

  // FIXME: move to model
  // Initialize runtime tensors
  model.x_ = Tensor(dim, Uninitialized<T>{});
  model.xb_ = Tensor(dim, Uninitialized<T>{});
  model.logits_ = Tensor(param.vocab_size, Uninitialized<T>{});
  model.scores_ = Tensor(dim, Uninitialized<T>{});

  for (int i = 0; i < n_layers; i++)
  {
    layers[i].key_cache_ =   Tensor({param.max_seq_len, dim}, T(0));
    layers[i].value_cache_ = Tensor({param.max_seq_len, dim}, T(0));
    layers[i].q_ =           Tensor(dim, Uninitialized<T>{});
  }
  // initialize ... ???
  //key_cache_ = Tensor2D(...);

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

//  h = 0 ... n_head:I
//    q = q_ + h * head_size
//    att = att_ + h * seq_len
//
//    t = 0 ... pos+1:
//      k = layer.key_cache + t * kv_dim + (h / kv_mul) * head_size
//
//      i = 0 ... head_size:
//      score += vecdot(q, k) // q {head_size} k {...}
//      att[t] = score/sqrt
//  
//    softmax(att, pos + 1) // att {pos+1}
//
// xb[...] = att.view(broadcast, x) * value_cache
//    xb = xb_ + h * head_sze
//    t = 0 ... pos + 1:
//      v = layer.value_cache + t * kv_dim + (h / kv_mul) * head_size
//      a = att[t]  // scores weight for this timestep
//
//      i = 0 ... head_size
//        xb[i] += a * v[i]; // += att[t] * v[i]
//  
//  xb2_ = xb_ * layer.wo   // {dim, dim} * {dim?, dim?} -> {dim, dim}

#if 0
  // FIXME need struct to handle these??
template <typename> unsigned int random_value(unsigned long long& state) return
{


unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}
#endif


#endif  // _LLAMA_H
