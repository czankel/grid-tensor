//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// Note: this file should not be included directly; it is included by llma.h

#ifndef _LLAMA_DEFAULT_H
#define _LLAMA_DEFAULT_H

namespace grid {


// FIXME: part of constructor or initialize?
template <typename T, typename Backend>
void LLaMAModelT<T,Backend>::Load()
{
  LLaMAModel::Parameters& params = file_->GetParameters();

  tok_embeddings_ = GetTensor2D("tok_embeddings.weight");
  norm_           = GetTensor2D("norm.weight");
  output_         = GetTensor2D("output.weight");

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


} // end of namespace grid

#endif  // _LLAMA_DEFAULT_H


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// Does T1.RmsNorm() perform in-place or return a copy?
// Options
//  - have two versions: RmsNorm() and RmsNorm_Inplace()
//     --> fuggly
//  - always return a "copy"
//     --> not optimal
//  - depend on type of tensor, if constant vs mutable
//     --> not really easily visible
//  - Might be assigned ot itself T1 = T1.RmsNorm() --> will call T1.RmsNorm()(T1)
//     --> requires something else to work, such as operator()(xxx)
//  - Have operator handle it: RmsNorm()() vs RmsNorm()(other)
//     --> seems the best option, also Tensor/TensorOp should have Temporary() to create or return the result tensor
//         That temporary might be the result or not?? unclear??
//     Not all operations require a copy of the buffer, i.e. shape, transpose, etc.
//  - Use move vs copy concepts
//     --> is this ugly?? I.e.
//          T1 + T2.RmsNorm()
//          T1 + std::move(T2.RmsNorm()) --> does that even work over RmsNorm(std::move...)
//     --> use better own macro with better name?
//          e.g.  T1 + InPlace(T2.RmsNorm())
//                T1 + Tensor::RmsNorm(std::move(T2))
//    --> Can RmsNorm be a move operator..hmmm would need both, so back to square one
//             T1 + RmsNormInplace(T2) === T1 + RmsNorm(std::move(T2))
//             T2.RmsNorm() vs RmsNorm(T2)
// - What about in-place add?
//     T1 + std::move(T2) or std::move(T1) + T2
//     Overwrite(T1) + T2
//     Overwrite(T1) + Overwrite(T2) ???
//     Unprotect(T1) + T2
//     Release(T1) + T2
//     InOut(T1) + T2
// --> std::move(Tensor) doesn't mean the buffer if available??? !!!
//
// --> Maybe const data is the way to go? 
//
//     T1 + T2 --> if either T1 or T2 is non-const data, use that buffer (if matches?)
//     --> doesn't work so well with add? 
//     (T1 + T2).RmsNorm() --> T1+T2 creates a mutable Tensor, RmsNorm will be in-place
//     T1.RmsNorm() --> if T1 is NNap --> typically const, unless mutable requested
//     --> What about reusing a tensor?
//          T1 = T2 + T3
//          T4 = T1.RmsNorm() --> T1 mutable, so in-place, even though assigned???
//          T5 = T1           --> wrong T1!!!
//     --> Assignment is non-in-place
//          T3 = (T1 + T2).RmsNorm()  --> T1 + T2 is not assigned, i.e. mutable
//          T4 = (T3 = T1 + T2).RmsNorm() --> T3 get's assigned to T4 --> non-mutable
//
//    --> How to implement?
//
//     T2.operator=(TensorOp T1) -->  Calls copy T1(out-of-place)
//     TensorOp(TensorOp T1) --> calls in-place T1(in-place) if non-const or out-of-placd
//     Tensor(TensorOp T1)   --> calls in-place T1(in-place) if non-const or out-of-place
//
//     Tensor& TensorOp::operator()()
//     Tensor TensorOp::operator()() const
//
//     --> dangerous?
//
//     Tensor t1 = ...;         --> in-place
//     Tensor t2 = t1.Op();     --> which one will it take?
//
//      --> operator=(const TensorOp&)  --> should take copy
//
//     Tensor t3 = TensorOp(t1.Op())
//     Tensor t3 = TensorOp(std::move(t1.Op()))
//
//      --> first is copy, second is in-place !!
//
//     t3 = t1 + t2.RmsNorm()  --> copy
//     t3 = t1 + std::move(t2.RmsNorm()) --> in-place
//
//     by using std::move, t2 cannot be used again, so the buffer __may__ be overwritten if not const
//
//     Differences?
//
//       t3 = t1 + std::move(t2.RmsNorm());   --> Add(const t1&, RmsNorm(t2)&&)
//       t3 = t1 + std::move(t2).RmsNorm();   --> Add(const t1&, t2.RmsNorm())
//
//         -->  Tensor::RmsNorm() const vs Tensor::RmsNorm()
//    --> Either works?
//
//     --> might be confusing for users?
//     --> Two options:
//          1. know how const works
//          2. use external functions:
//              RmsNorm(Tensor) --> copy
//              Add(Tensor, Tensor) --> copy
//              Tensor + Tensor --> copy unless :
//              std::move(Tensor) + Tensor --> in-place
//
//      result = tensor1 + std::move(tensor2);
//      --> result.operator=(Add(const Tensor&, Tensor&&))
//
//      if constexpr (is_rvalue_reference_v<Tensor> && is_writable<Tensor>)) // && dimensions correct, and no strides, etc.?
//      result = Tensor;
//      else
//      result = new Tensor()
//
//
//
//  --> Tensor&& doesn't mean it can be overwritten??
//
//
//
//
//
//
//     T1 * T2.Transpose() --> shouldn't change T2 for future .. 
//     Options:
//      - return a copy of the tensor
//      - return a TensorOp
//          --> Add(..) will call TensorTranspose() which will return a copy so same as first option?
// --> Returning a Tensor copy is not the problem, duplicating/copying the underlying buffer is
// --> Any operation that results in an operation on data should return a TensorOp, other operations can immediately return a Tensor
//
// TR = T1 + T2 + T3 + T4 ...
// TR.operator=(TensorAdd(... TensorAdd(TensorAdd(T1, T2), T3), ...)
//                 add_1_n       add_1_3           add_1_2
//
// add_1_n(result)
// --> might not be Add always, something could change the result size (pooling)
// --> in general, a result-tensor in a Tensor might not always have the right size
// --> move memory handling outside, while minimizing overall memory allocation, i.e. release buffers quickly
// --> T1.RmsNorm()  
//
//  Note: operator()(target) can get tricky?
//    Add(ConvertibleTensor, ConvertibleTensor)
//    {
//      if (T1 = TensorOp && resultbuffer)
//        resultbuffer = T1.resultbuffer
//      else if (T2 = TensorOp && has_resultbuffer)
//        ...
//      else
//        result = new Tensor()
//    }
//    
//    operator()(result)
//    {
//      if (T1 = TensorOp)
//        T1(result)
//      
//  can't really re-use result buffer between two  operations as each (can or does?) destroy the result buffer
//  However, Add(TensorOp, Tensor) .. or Tensor,TensorOp can use single result buffer
//   who would be best to manage??? caller? except, might not now dimensions?? also tensor might already have a result buffer?
//
//   Tensor T = some operations that result in a result buffer
//
//   Add(T, xx)
//
//   might not be optimal??? i.e. padding?? ... should be marked internally i..e not expose the result buffer
//   GetResultBuffer() { if !optimal return nullptr } ... better name, is tensor ... ResultTensor() or Result() or Ephemeral() or Termporary()
//
//  Add(ConvertibleTensor, ...)
//    if (T1 = TensorOp)
//      result = T1.Placeholder()


#if 0
template <typename T, typename Comp>
void LLaMAModel::Forward(
    const llama_token* tokens,
    const int ntokens,
    const int nports)
{

  Tensor<int32_t,B> embedded({ntokens}, tokens)
  //memcpy(embd->data, tokens, ntokens x elem_size(embd));

  Tensor input = TensorProduct(tok_embeddings_, tokens);
  std::cout << input;

  inpl = get_rows(model.tok_embeddings, embd);  // ?? is this a Rank-1?

  for (int il = 0; il < nlayers; il++)
  {
    Tensor* inpSA = inpl;
    cur = Tensor::RmsNorm(inpl);

    // cur = cur*attention_norm(broadcasted)
    cur = mul(cur, model.layers[il].attention_norm);  // Rank X x Rank 2   ??

    // self-attention
    // compute q and k and RoPE them
    Tensor& tmpk = Tensor::MatMul(model.layers[il].wk, cur);
    Tensor& tmpq = Tensor::MatMul(mdoel.layers[il].wq, cur);

    Tensor& kCur = rope_inplace(reshape_3d(tmpk, nembedding/nheads, nheads, N), n_past, n_rot, 0, 0);
    Tensor& Qcur = rope_inplace(reshape_3d(tmpq, nembd/nhead, nhead, N), n_post, n_rot, 0, 0);

    // store key and value to memory
    // compute the transposed [N, n_embd] V matrix
    Tensor& tmpv = MatMul(model.layers[i].wv, cur);
    Tensor& Vcur = transpose(reshape_2d(tmpv, n_mebd, N));

    Tensor& k = view_1d(kv_self.k, N * n_mebd, (....));
    Tensor& v = view_2d(kv_selv.v, N, n_embd, ...);

    // important, storing RoPEed version of K in the KV cache!
    build_forward_expand(&gf, cpy(Kcur, k));
    build_forward_expand(&gf, cpy(Vcur, v));

    Tensor& Q = permute(Qcur, 0, 2, 1, 3);
    Tensor& K = permute(reshape_3d(view_1d(kv_self.k, (n_past + N)*n_embd, il*n_ctx*element_size...)));

    // K + Q
    Tensor& KQ = MatMul(K, Q);
    Tensor& KQ_scale = new_f32(1.0f/sqrtf(float(n_embd)/n_head));
    Tensor& KQ_scaled = scale_inplace(KQ, KQ_scale);
    Tensor& KQ_masked = diag_mask_inf_inplace(KQ_scaled, n_post);
    Tensor& KQ_soft_max = soft_max_inplace(KQ_masked);
  Tensor& V = view_3d(kv_self.v, n_past + N, n_embd/n_head, n_head, n_ctx...);

  Tensor& KQV = MatMul(V, KQ_soft_max);

  Tensor& KQV_merged = permute(KQV, 0, 2, 1, 3);

  cur = cpy(KQV_merged, new_tensor_2d(f32_t, n_mebd, N));
  cur = MatMul(model.layers[il].wo, cur);
  }
};
#endif

