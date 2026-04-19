#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include <base/tick.h>
namespace kernel {
constexpr static int thread_num = 256;
constexpr static int max_kv_group_size = 16;
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  // this should be FLT_MAX, not 0 !!!!
  // otherwise, the softmax may be occur nan when head_dim < 128 threads
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

__global__ void multi_head_attention_kernel_legacy(int32_t pos, int32_t seq_len, float* query,
                                                   float* score_ptr, float* output,
                                                   float* key_cache, float* value_cache,
                                                   int32_t kv_dim, int32_t kv_mul,
                                                   int32_t head_num, int32_t head_size,
                                                   int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

    float score = 0.0f;
    int i = 0;
    for (; i + 3 < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);

      score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
    }
    for (; i < head_size; ++i) {
      score += key_head[i] * s_query_head[i];
    }

    score_head[t] = score * scale;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      value += score_head[t] * value_head[i];
    }
    output_head[i] = value;
  }
}


__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset) {
  int kv_head_num = head_num / kv_mul;
  int kv_head = blockIdx.x;
  if (kv_head >= kv_head_num) {
    return;
  }

  extern __shared__ float s_query_heads[];
  float scale = 1.f / sqrtf(float(head_size));
  int q_head_begin = kv_head * kv_mul;
  int head_offset = kv_head * head_size;

  // 预加载当前kv group里的所有query到共享内存
  int query_group_size = kv_mul * head_size;
  for (int i = threadIdx.x; i < query_group_size; i += blockDim.x) {
    s_query_heads[i] = query[q_head_begin * head_size + i];
  }
  __syncthreads();

  // 每个时间步只加载一次共享的key，然后为组内所有query head计算分数
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
    for (int local_q = 0; local_q < kv_mul; ++local_q) {
      float* query_head = s_query_heads + local_q * head_size;
      float* score_head = score_ptr + (q_head_begin + local_q) * seq_len;

      float score = 0.0f;
      int i = 0;
      for (; i + 3 < head_size; i += 4) {
        float4 key_val = *reinterpret_cast<float4*>(key_head + i);
        float4 query_val = *reinterpret_cast<float4*>(query_head + i);
        score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
                 key_val.w * query_val.w;
      }
      for (; i < head_size; ++i) {
        score += key_head[i] * query_head[i];
      }

      score_head[t] = score * scale;
    }
  }
  __syncthreads();

  for (int local_q = 0; local_q < kv_mul; ++local_q) {
    float* score_head = score_ptr + (q_head_begin + local_q) * seq_len;
    softmax_gpu(score_head, pos + 1);
    __syncthreads();
  }

  // 复用同一份value读取，为组内所有query head做加权求和
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float values[max_kv_group_size] = {0.0f};
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float value = value_head[i];
      for (int local_q = 0; local_q < kv_mul; ++local_q) {
        float* score_head = score_ptr + (q_head_begin + local_q) * seq_len;
        values[local_q] += score_head[t] * value;
      }
    }

    for (int local_q = 0; local_q < kv_mul; ++local_q) {
      float* output_head = output + (q_head_begin + local_q) * head_size;
      output_head[i] = values[local_q];
    }
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  if (kv_mul > 1 && kv_mul <= max_kv_group_size && head_num % kv_mul == 0) {
    int32_t kv_head_num = head_num / kv_mul;
    size_t shared_mem_size = static_cast<size_t>(kv_mul) * head_size * sizeof(float);
    multi_head_attention_kernel<<<kv_head_num, thread_num, shared_mem_size, stream>>>(
        pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
        head_size, layer_offset);
    return;
  }

  multi_head_attention_kernel_legacy<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
}

}  // namespace kernel
