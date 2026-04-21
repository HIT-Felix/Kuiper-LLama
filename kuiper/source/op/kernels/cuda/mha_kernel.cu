#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "base/profiler.h"
#include "mha_kernel.cuh"
#include <base/tick.h>
namespace kernel {
constexpr static int thread_num = 256;

__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, const int32_t* block_table,
                                            int32_t block_size, int32_t block_num, int32_t kv_dim,
                                            int32_t kv_mul, int32_t head_num, int32_t head_size,
                                            int32_t layer_index) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float shared_storage[];
  float* s_query_head = shared_storage;
  float* s_output_acc = shared_storage + head_size;
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float s_running_max;
  __shared__ float s_running_denom;
  __shared__ float s_prev_scale;
  __shared__ float s_token_scale;

  UNUSED(score_ptr);
  const bool use_paged_kv = block_table != nullptr && block_size > 0;
  const int32_t block_stride = block_size * kv_dim;
  const int32_t layer_stride = block_num * block_stride;
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  // 预加载query到共享内存
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
    s_output_acc[i] = 0.f;
  }
  if (threadIdx.x == 0) {
    s_running_max = -FLT_MAX;
    s_running_denom = 0.f;
  }
  __syncthreads();

  // head当前的注意力头索引，kv_mul用于gqa，head_size表示一个自注意力头的维度
  // kv_dim = head_size * head_num，多头自注意力情况下的key,value 维度
  // kv_dim = head_size * head_num / kv_num，GQA情况下的key,value 维度
  int head_offset = (head / kv_mul) * head_size;
  float* output_head = output + head * head_size;

  float* current_key_block = nullptr;
  float* current_value_block = nullptr;
  if (!use_paged_kv) {
    current_key_block = key_cache + layer_index * seq_len * kv_dim;
    current_value_block = value_cache + layer_index * seq_len * kv_dim;
  }

  for (int t = 0; t <= pos; ++t) {
    if (use_paged_kv && t % block_size == 0) {
      const int32_t logical_block_idx = t / block_size;
      const int32_t physical_block_idx = block_table[logical_block_idx];
      current_key_block =
          key_cache + layer_index * layer_stride + physical_block_idx * block_stride;
      current_value_block =
          value_cache + layer_index * layer_stride + physical_block_idx * block_stride;
    }

    float* key_head = nullptr;
    float* value_head = nullptr;
    if (use_paged_kv) {
      const int32_t token_offset = t % block_size;
      key_head = current_key_block + token_offset * kv_dim + head_offset;
      value_head = current_value_block + token_offset * kv_dim + head_offset;
    } else {
      key_head = current_key_block + t * kv_dim + head_offset;
      value_head = current_value_block + t * kv_dim + head_offset;
    }

    float thread_score = 0.0f;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
      thread_score += s_query_head[i] * key_head[i];
    }

    float score = BlockReduce(reduce_storage).Sum(thread_score) * scale;
    if (threadIdx.x == 0) {
      const float new_running_max = fmaxf(s_running_max, score);
      const float prev_scale = expf(s_running_max - new_running_max);
      const float token_scale = expf(score - new_running_max);
      s_prev_scale = prev_scale;
      s_token_scale = token_scale;
      s_running_denom = s_running_denom * prev_scale + token_scale;
      s_running_max = new_running_max;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
      s_output_acc[i] = s_output_acc[i] * s_prev_scale + s_token_scale * value_head[i];
    }
    __syncthreads();
  }

  const float inv_denom = 1.f / s_running_denom;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    output_head[i] = s_output_acc[i] * inv_denom;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, int32_t block_size,
                   const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                   const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                   const tensor::Tensor& value_cache_tensor,
                   const tensor::Tensor& block_table_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());
  const int32_t* block_table =
      block_table_tensor.is_empty() ? nullptr : block_table_tensor.ptr<int32_t>();
  const int32_t block_num = block_table_tensor.is_empty() ? 0 : block_table_tensor.get_dim(0);
  float* score = block_table ? nullptr : const_cast<float*>(score_tensor.ptr<float>());
  base::ScopedCudaProfile profile(block_table ? "PagedMHA" : "ContiguousMHA",
                                  config ? config->stream : nullptr);

  cudaStream_t stream = config->stream;
  multi_head_attention_kernel<<<head_num, thread_num, 2 * head_size * sizeof(float), stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, block_table, block_size,
      block_num, kv_dim, kv_mul, head_num, head_size, layer_index);
}

}  // namespace kernel
