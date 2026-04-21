#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {

struct LLama2Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> qkv_layers_;
  std::vector<tensor::Tensor> qkv_weights_;
  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

struct LLamaGenerationStats {
  double prefill_latency_ms = 0.0;
  double decode_latency_ms = 0.0;
  double ttft_ms = 0.0;
  int32_t prefill_tokens = 0;
  int32_t decode_tokens = 0;
  int32_t allocated_blocks = 0;
  int32_t block_size = 0;
  int32_t prompt_last_block_tokens = 0;
  double prompt_tail_fragmentation = 0.0;
  size_t kv_bytes_used = 0;
  size_t kv_bytes_reserved = 0;
};

class LLama2Model : public Model {
 public:
  explicit LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status prefill(const std::vector<int32_t>& prompt_tokens, int32_t& next) const;

  base::Status decode_step(int32_t token, int32_t pos, int32_t& next) const;

  void reset_generation_state() const;

  const LLamaGenerationStats& generation_stats() const;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                           int32_t token_pos) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

  base::Status prepare_paged_blocks_for_pos(int32_t pos) const;

  void sync_stream() const;

  void log_prefill_summary() const;

  void log_decode_progress(int32_t pos, int32_t token) const;

private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<LLama2Layers> llama_layers_;
  int32_t paged_kv_block_size_ = 16;
  int32_t paged_kv_block_num_ = 0;
  mutable int32_t active_paged_blocks_ = 0;
  mutable LLamaGenerationStats generation_stats_;
};
}  // namespace model

#endif
