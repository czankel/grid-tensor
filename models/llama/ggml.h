//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_LLAMA_GGML_H
#define GRID_LLAMA_GGML_H

#include <vector>

#include <grid/models/llama.h>
#include <grid/tensor/mmap.h>
#include <grid/tensor/tensor.h>

#include "llama.h"

namespace grid {


enum GgmlMagic
{
  kGgmlMagicGGJT = 0x67676a74u, // 'ggjt'
  kGgmlMagicGGLA = 0x67676c61u, // 'ggla'
  kGgmlMagicGGMF = 0x67676d66u, // 'ggmf'
  kGgmlMagicGGML = 0x67676d6cu, // 'ggml'
  kGgmlMagicGGSN = 0x6767736eu, // 'ggsn'
};


enum GgmlDataType
{
  kGgmlDataTypeInvalid = -1,
  kGgmlDataTypeF32  =  0,
  kGgmlDataTypeF16  =  1,
  kGgmlDataTypeQ4_0 =  2,
  kGgmlDataTypeQ4_1 =  3,
  kGgmlDataTypeQ5_0 =  6,
  kGgmlDataTypeQ5_1 =  7,
  kGgmlDataTypeQ8_0 =  8,
  kGgmlDataTypeQ8_1 =  9,
  kGgmlDataTypeQ2_K = 10,
  kGgmlDataTypeQ3_K = 11,
  kGgmlDataTypeQ4_K = 12,
  kGgmlDataTypeQ5_K = 13,
  kGgmlDataTypeQ6_K = 14,
  kGgmlDataTypeQ8_K = 15,
  kGgmlDataTypeI8,
  kGgmlDataTypeI16,
  kGgmlDataTypeI32,
  kGgmlDataTypeCount,
};


const size_t kQuantsQ4_0 = 32;
struct BlockQ4_0
{
  float16_t delta;
  uint8_t qs[kQuantsQ4_0 / 2];
};

const size_t kQuantsQ4_1 = 32;
struct BlockQ4_1
{
  float16_t delta;
  float16_t min;
  uint8_t qs[kQuantsQ4_1 / 2];
};

const size_t kQuantsQ5_0 = 32;
struct BlockQ5_0
{
  float16_t delta;
  uint8_t qh[4];          // 5th bit of quants
  uint8_t qs[kQuantsQ5_0 / 2];
};

const size_t kQuantsQ5_1 = 32;
struct BlockQ5_1 {
  float16_t delta;
  float16_t min;
  uint8_t qh[4];          // 5th bit of quants
  uint8_t qs[kQuantsQ5_0 / 2];
};

const size_t kQuantsQ8_0 = 32;
struct BlockQ8_0
{
  float16_t delta;
  uint8_t qs[kQuantsQ8_0];
};

const size_t kQuantsQ8_1 = 32;
struct BlockQ8_1 {
  float16_t delta;
  float16_t min;
  uint8_t qs[kQuantsQ8_1];
};


// TODO: use C++20 initializers
static const size_t TypeSize[kGgmlDataTypeCount] =
{
  /* kF32 */  sizeof(float),
  /* kF16 */  sizeof(float16_t),
  /* kQ4_0 */ sizeof(BlockQ4_0),
  // FIXME: complete table
};


// Note that UNKNOWN is -1
static const GgmlDataType GgmlFileToDataType[] =
{
  /*  0: ALL_F32     0          */  kGgmlDataTypeF32,
  /*  1: MOSTLY_F16  1          */  kGgmlDataTypeF16,
  /*  2: MOSTLY_Q4_0 2          */  kGgmlDataTypeQ4_0,
  /*  3: MOSTLY_Q4_1 3          */  kGgmlDataTypeQ4_1,
  /*  4: MOSTLY_Q4_1_SOME_F16 4 */  kGgmlDataTypeInvalid,
  /*  5: invalid                */  kGgmlDataTypeInvalid,
  /*  6: invalid                */  kGgmlDataTypeInvalid,
  /*  7: MOSTLY_Q8_0 7          */  kGgmlDataTypeQ8_0,
  /*  8: MOSTLY_Q5_0 8          */  kGgmlDataTypeQ5_0,
  /*  9: MOSTLY_Q5_1 9          */  kGgmlDataTypeQ5_1,
  /* 10: MOSTLY_Q2_K 10         */  kGgmlDataTypeQ2_K,
  /* 11: MOSTLY_Q3_K 11         */  kGgmlDataTypeQ3_K,
  /* 12: MOSTLY_Q4_K 12         */  kGgmlDataTypeQ4_K,
  /* 13: MOSTLY_Q5_K 13         */  kGgmlDataTypeQ5_K,
  /* 14: MOSTLY_Q6_K 14         */  kGgmlDataTypeQ6_K,
};


const size_t kQuantsQK_K = 0;  // not implemented
static const size_t QuantSize[kGgmlDataTypeCount] =
{
  /* kGgmlDataTypeF32           */  1,
  /* kGgmlDataTypeF16           */  1,
  /* kGgmlDataTypeQ4_0          */  kQuantsQ4_0,
  /* kGgmlDataTypeQ4_1          */  kQuantsQ4_1,
  /* unused                     */  0,
  /* unused                     */  0,
  /* kGgmlDataTypeQ5_0          */  kQuantsQ5_0,
  /* kGgmlDataTypeQ5_1          */  kQuantsQ5_1,
  /* kGgmlDataTypeQ8_0          */  kQuantsQ8_0,
  /* kGgmlDataTypeQ8_1          */  kQuantsQ8_1,
  /* kGgmlDataTypeQ2_K          */  kQuantsQK_K,
  /* kGgmlDataTypeQ3_K          */  kQuantsQK_K,
  /* kGgmlDataTypeQ4_K          */  kQuantsQK_K,
  /* kGgmlDataTypeQ5_K          */  kQuantsQK_K,
  /* kGgmlDataTypeQ6_K          */  kQuantsQK_K,
  /* kGgmlDataTypeQ8_K          */  kQuantsQK_K,
  /* kGgmlDataTypeI8            */  1,
  /* kGgmlDataTypeI16           */  1,
  /* kGgmlDataTypeI32           */  1,
};
/// TensorShard is one shard with pointer and size.
struct TensorShard
{
  void*  address_;
  size_t size_;
};



// GgmlModelParams is included in the ggml file and holds model information.
struct GgmlModelParams
{
  uint32_t vocab_size_          = 32000;  // FIXME n_vocab
  uint32_t hidden_size_         = 4096;   // FIXME n_embd or ?? // isn't this just embedded_? // is this dimension of embeddings or same as vocab?? somehoe?
  uint32_t mult_                = 256;    // FIXME mult is result of find_n_mult(intermediate_size, hidden_size)
  uint32_t num_attention_heads_ = 32;     // n_head
  uint32_t num_hidden_layers_   = 32;     // FIXME n_layer
  uint32_t obsolete_            = 64;     // obsolete in some?? n_rot nembd / n_head
  GgmlDataType ftype_;                    // Note that the stored file-type differs from data-type
};
  // FIXME: FileTensor is internal... should have way to get actual tensor? Make this virtual protected and provide actual GetTensor<TYPE>?
  //virtual FileTensor* GetFileTensor(const std::string& name, const std::type_info& type) = 0;


#if 0
  // FIXME:  likely move out? Isn't this only ggml? for now, at least?
  /// FileTensor describes a particular tensor array in the file.
  ///
  /// Tensors are 'sharded' in different files or section for larger models.
  /// The smaller tensors are duplicated in each section.
  struct FileTensor
  {
    FileTensor(const std::type_info& type_info)
      : type_info_(type_info),
        transposed_(false),
        rank_(0),
        shard_count_(0)
    {}

    const std::type_info& type_info_;

    bool        transposed_;
    uint32_t    rank_;
    uint32_t    dims_[kMaxTensorRank];
    uint32_t    shard_count_;
    TensorShard shards_[kMaxTensorShards];
  };
#endif

/// GgmlFileTensor describes a particular tensor and offset in the file.
struct GgmlFileTensor : LLaMAFile::FileTensor
{
  GgmlFileTensor(const std::type_info& type_info) : FileTensor(type_info) {}
  GgmlDataType dtype_;
};


// GgmlFile handles files generated with the ggml conversion tool.
class GgmlFile : public LLaMAFile
{
  const std::type_info& DataType(GgmlDataType dtype);

 public:
  virtual ~GgmlFile() = default;

  virtual const std::type_info& DataType();
  virtual LLaMAFile::FileTensor* GetFileTensor(const std::string& name, const std::type_info& dtype);
  virtual LLaMAModel::Parameters& GetParameters() { return params_; }

  static bool IsSupported(const std::string& path);
  static LLaMAFile* Open(const std::string& path, bool mmap);

 private:
  GgmlFile(MMap&& mmap) : mmap_(mmap) {}
  void Load();

 private:
  uint32_t        magic_;
  uint32_t        version_;
  GgmlDataType    ftype_;
  MMap            mmap_;

  LLaMAModel::Parameters  params_;
  LLaMAVocab              vocab_;

  std::unordered_map<std::string, GgmlFileTensor> tensors_;
};


} // end of namespace grid

#endif // GRID_LLAMA_GGML_H


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


#if 0
  std::vector<GgmlFileTensor> tensors_;
  std::unordered_map<std::string, size_t> tensor_idx_;
// being refactored out
enum GgmlFileType
{
  kAllF32              = 0,
  kMostlyF16           = 1,
  kMostlyQ4_0          = 2,
  kMostlyQ4_1          = 3,
  kMostlyQ4_1SomeF16   = 4, // tok_embeddings.weight and output.weight are f16
  kMostlyQ8_0          = 7,
  kMostlyQ5_0          = 8,
  kMostlyQ5_1          = 9,
  kMostlyQ2_k          = 10,
  kMostlyQ3_k_s        = 11,
  kMostlyQ3_k_m        = 12,
  kMostlyQ3_k_l        = 13,
  kMostlyQ4_k_s        = 14,
  kMostlyQ4_k_m        = 15,
  kMostlyQ5_k_s        = 16,
  kMostlyQ5_k_m        = 17,
  kMostlyQ6_k          = 18,
};
#endif
