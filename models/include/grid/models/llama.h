//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_MODEL_LLAMA_H
#define GRID_MODEL_LLAMA_H

#include <grid/tensor/tensor.h>

#include <iostream>

namespace grid {

const size_t kMaxTensorRank = 2;
const size_t kMaxTensorShards = 1;

template <template <typename, size_t, auto...> typename, typename, auto...> class LLaMAModelT;
class LLaMAFile;

/// LLaMAModel is an interface for providing a LLaMA base class without any templated paramters.
class LLaMAModel
{
 public:
  struct Parameters
  {
    size_t vocab_size_;
    size_t hidden_size_;
    size_t num_hidden_layers_;
    size_t num_attention_heads_;
  };

  static constexpr uint32_t kBOS = 1;
  static constexpr uint32_t kEOS = 2;

 public:
  virtual ~LLaMAModel() = default;

  /// FIXME
  virtual void Generate(const std::string&, int steps) = 0;

  /// Load creates a new model and loads the specified file or memory-maps the file.
  template <template <typename, size_t, auto...> typename Tensor>
  static LLaMAModel* Load(const LLaMAFile&, bool mmap = true);

  /// Create creates a new model in memory.
  template <template <typename, size_t, auto...> typename Tensor, typename T>
  static LLaMAModel* Create(Parameters&);

  /// Create creates a new model for a memory mapped file.
  template <template <typename, size_t, auto...> typename Tensor, typename T>
  static LLaMAModel* Create(LLaMAFile&, Parameters&);
};


/// LLaMAFile is an interface for managing LLaMA files.
class LLaMAFile
{
 public:
  virtual ~LLaMAFile() = default;

  /// DataType returns the dominant data type of the file.
  virtual const std::type_info& DataType() const = 0;

  /// GetParameters returns a reference to the model paramters.
  virtual LLaMAModel::Parameters GetParameters() const = 0;

  /// PrintModelParameters prints the model informatio and parameter
  void PrintModelInfo(std::ostream&) const;
};


/// KarpathyFile are snapshots created by the Karpathy LLaMA2 implementation.
/// https://github.com/karpathy/llama2.c
class KarpathyFile : public LLaMAFile
{
  // FIXME
  // - sample: sequence of words/learn case
  // - embeddings: meaning of words
  // - encoder: typically 6 encoders / layers / ...
  // - multi-head attention: parallel attentions combined in a merge step
  // - input sequence: 1...full sequence (are these 'tokens'?)
  // - embeddings [sequence/position, embeddings-dimension]
  // - attention: query, key, value: n_heads x [
  // Query Size = Embedding Size / Number of heads
  // FIXME: aren't these uint32s?
  struct FileParameters
  {
    int dim;          // transformer dimension
    int hidden_dim;   // numberof ffn dimensions
    int n_layers;     // number of layers
    int n_heads;      // number of query heads
    int n_kv_heads;   // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;   // vocabulary size, usually 256 (byte-level) // FIXME: same as max_token_...?
    int seq_len;      // max sequence length
  };

 public:
  /// Constructor FIXME
  /// @throws system_error if the file does not exist.
  KarpathyFile(const std::string& path, const std::string& tokenizer_path = "tokenizer.bin");
  virtual ~KarpathyFile() = default;

  virtual const std::type_info& DataType() const;
  virtual LLaMAModel::Parameters GetParameters() const;

  /// Load initiaalizes the model with memory-mapped FIXME
  template <template <typename, size_t, auto...> typename Tensor, typename T, auto... Args>
  void Load(LLaMAModelT<Tensor, T, Args...>&, const LLaMAFile&) const;

 private:
  std::string     tokenizer_path_;
  std::string     path_;
  size_t          file_size_;
  FileParameters  parameters_;
};


} // end namespace grid

#endif // GRID_MODEL_LLAMA_H
