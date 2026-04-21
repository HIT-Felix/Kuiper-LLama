#include "../cpu/mha_kernel.h"
#include <cuda_runtime_api.h>
#include "../kernels_interface.h"
namespace kernel {
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, int32_t block_size,
                const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                const tensor::Tensor& value_cache_tensor, const tensor::Tensor& block_table_tensor,
                base::DeviceType device_type, CudaConfig* config) {
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));
  const bool use_paged_kv = block_size > 0 && !block_table_tensor.is_empty();

  std::shared_ptr<base::DeviceAllocator> allocator;
  if (device_type == base::DeviceType::kDeviceCPU) {
    allocator = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    allocator = base::CUDADeviceAllocatorFactory::get_instance();
  }

  const int32_t block_num = use_paged_kv ? key_cache_tensor.get_dim(1) : 0;
  const int32_t block_stride = block_size * kv_dim;
  const int32_t layer_stride = block_num * block_stride;
  for (int32_t h = 0; h < head_num; ++h) {
    float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
    float* query_head_addr = const_cast<float*>(query_tensor.ptr<float>() + h * head_size);

    
    tensor::Tensor query_mat(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                               query_head_addr);
    query_mat.set_device_type(device_type);
    
    for (int32_t t = 0; t <= pos; t++) {
      const int32_t head_offset = (h / kv_mul) * head_size;
      int32_t cache_offset = 0;
      if (use_paged_kv) {
        const int32_t logical_block_idx = t / block_size;
        const int32_t token_offset = t % block_size;
        CHECK_LT(logical_block_idx, block_table_tensor.size());
        const int32_t physical_block_idx = block_table_tensor.index<int32_t>(logical_block_idx);
        cache_offset = layer_index * layer_stride + physical_block_idx * block_stride +
                       token_offset * kv_dim + head_offset;
      } else {
        cache_offset = layer_index * seq_len * kv_dim + t * kv_dim + head_offset;
      }
      const float* key_head_addr = key_cache_tensor.ptr<float>() + cache_offset;
      tensor::Tensor key_mat(base::DataType::kDataTypeFp32, 1, head_size, false, nullptr,
                             const_cast<float*>(key_head_addr));
      
      tensor::Tensor score_mat(base::DataType::kDataTypeFp32, 1, false, nullptr,
                               score_head_addr + t);
      key_mat.set_device_type(device_type);
      score_mat.set_device_type(device_type);
      get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
    }

    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, pos + 1, false, nullptr,
                                     score_head_addr);
    score_head_tensor.set_device_type(device_type);
    get_softmax_kernel(device_type)(score_head_tensor, config ? config->stream : nullptr);

    float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
    allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                              config ? config->stream : nullptr, false);
    tensor::Tensor output_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                 output_head_ptr);
    output_tensor.set_device_type(device_type);

    if (!use_paged_kv) {
      const int32_t cache_offset = layer_index * seq_len * kv_dim + (h / kv_mul) * head_size;
      float* value_head_addr =
          const_cast<float*>(value_cache_tensor.ptr<float>()) + cache_offset;
      tensor::Tensor value_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                  value_head_addr);
      get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor, output_tensor, pos,
                                        head_size, kv_dim, config ? config->stream : nullptr);
      continue;
    }

    for (int32_t i = 0; i < head_size; ++i) {
      float value = 0.f;
      for (int32_t t = 0; t <= pos; ++t) {
        const int32_t logical_block_idx = t / block_size;
        const int32_t token_offset = t % block_size;
        const int32_t physical_block_idx = block_table_tensor.index<int32_t>(logical_block_idx);
        const int32_t cache_offset = layer_index * layer_stride + physical_block_idx * block_stride +
                                     token_offset * kv_dim + (h / kv_mul) * head_size + i;
        value += score_head_addr[t] * value_cache_tensor.index<float>(cache_offset);
      }
      output_head_ptr[i] = value;
    }
  }
}
}  // namespace kernel
