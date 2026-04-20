#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32_basic(const float* input, const float* weight, float* output,
                                            int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK, int TILE_K>
__global__ void matmul_kernel_cu_fp32_tiled(const float* input, const float* weight, float* output,
                                            int M, int K) {
  const unsigned int tid = threadIdx.x;
  const int start_row = blockIdx.x * ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  __shared__ float input_tile[TILE_K];
  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage reduce_storage[ROW_PER_BLOCK];

  float partial_sums[ROW_PER_BLOCK];
#pragma unroll
  for (int r = 0; r < ROW_PER_BLOCK; ++r) {
    partial_sums[r] = 0.f;
  }

  for (int tile_start = 0; tile_start < M; tile_start += TILE_K) {
    const int tile_size = min(TILE_K, M - tile_start);

    for (int i = tid; i < tile_size; i += THREAD_PER_BLOCK) {
      input_tile[i] = input[tile_start + i];
    }
    __syncthreads();

#pragma unroll
    for (int r = 0; r < ROW_PER_BLOCK; ++r) {
      const int row = start_row + r;
      if (row >= K) {
        continue;
      }

      const float* row_weight = weight + row * M + tile_start;
      float local_sum = 0.f;

      for (int i = tid; i < tile_size; i += THREAD_PER_BLOCK) {
        local_sum += input_tile[i] * row_weight[i];
      }
      partial_sums[r] += local_sum;
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < ROW_PER_BLOCK; ++r) {
    const int row = start_row + r;
    if (row >= K) {
      continue;
    }

    const float row_sum = BlockReduce(reduce_storage[r]).Sum(partial_sums[r]);
    __syncthreads();
    if (tid == 0) {
      output[row] = row_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK, int TILE_K>
__global__ void matmul_kernel_cu_fp32_lm_head(const float* input, const float* weight,
                                              float* output, int M, int K) {
  static_assert(TILE_K % 4 == 0);

  const unsigned int tid = threadIdx.x;
  const int start_row = blockIdx.x * ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int kVecWidth = 4;
  constexpr int kVecTile = TILE_K / kVecWidth;
  __shared__ float4 input_tile[kVecTile];
  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage reduce_storage[ROW_PER_BLOCK];

  float partial_sums[ROW_PER_BLOCK];
#pragma unroll
  for (int r = 0; r < ROW_PER_BLOCK; ++r) {
    partial_sums[r] = 0.f;
  }

  for (int tile_start = 0; tile_start < M; tile_start += TILE_K) {
    const float4* input_vec = reinterpret_cast<const float4*>(input + tile_start);
    for (int i = tid; i < kVecTile; i += THREAD_PER_BLOCK) {
      input_tile[i] = input_vec[i];
    }
    __syncthreads();

#pragma unroll
    for (int r = 0; r < ROW_PER_BLOCK; ++r) {
      const int row = start_row + r;
      if (row >= K) {
        continue;
      }

      const float4* row_weight =
          reinterpret_cast<const float4*>(weight + row * M + tile_start);
      float local_sum = 0.f;
      for (int i = tid; i < kVecTile; i += THREAD_PER_BLOCK) {
        const float4 input4 = input_tile[i];
        const float4 weight4 = row_weight[i];
        local_sum += input4.x * weight4.x + input4.y * weight4.y + input4.z * weight4.z +
                     input4.w * weight4.w;
      }
      partial_sums[r] += local_sum;
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < ROW_PER_BLOCK; ++r) {
    const int row = start_row + r;
    if (row >= K) {
      continue;
    }

    const float row_sum = BlockReduce(reduce_storage[r]).Sum(partial_sums[r]);
    __syncthreads();
    if (tid == 0) {
      output[row] = row_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  CHECK_EQ(M, input.get_dim(0));
  constexpr int kThreadsPerBlock = 128;
  constexpr int kRowsPerBlock = 4;
  constexpr int kTileK = 256;
  constexpr int kLmHeadRowsPerBlock = 8;
  constexpr int kLmHeadTileK = 512;
  const bool is_lm_head_shape = (M == 3072 && K >= 65536 && M % kLmHeadTileK == 0);
  const int32_t block_num = (K + kRowsPerBlock - 1) / kRowsPerBlock;
  const int32_t lm_head_block_num = (K + kLmHeadRowsPerBlock - 1) / kLmHeadRowsPerBlock;

  if (config && config->stream) {
    if (is_lm_head_shape) {
      matmul_kernel_cu_fp32_lm_head<kThreadsPerBlock, kLmHeadRowsPerBlock, kLmHeadTileK>
          <<<lm_head_block_num, kThreadsPerBlock, 0, config->stream>>>(
              input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M,
              K);
    } else if (M >= kTileK) {
      matmul_kernel_cu_fp32_tiled<kThreadsPerBlock, kRowsPerBlock, kTileK>
          <<<block_num, kThreadsPerBlock, 0, config->stream>>>(
              input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M,
              K);
    } else {
      matmul_kernel_cu_fp32_basic<kThreadsPerBlock, 1>
          <<<K, kThreadsPerBlock, 0, config->stream>>>(
              input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M,
              K);
    }
  } else {
    if (is_lm_head_shape) {
      matmul_kernel_cu_fp32_lm_head<kThreadsPerBlock, kLmHeadRowsPerBlock, kLmHeadTileK>
          <<<lm_head_block_num, kThreadsPerBlock>>>(
              input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M,
              K);
    } else if (M >= kTileK) {
      matmul_kernel_cu_fp32_tiled<kThreadsPerBlock, kRowsPerBlock, kTileK>
          <<<block_num, kThreadsPerBlock>>>(
              input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M,
              K);
    } else {
      matmul_kernel_cu_fp32_basic<kThreadsPerBlock, 1>
          <<<K, kThreadsPerBlock>>>(input.ptr<float>(), weight.ptr<float>(),
                                    const_cast<float*>(output.ptr<float>()), M, K);
    }
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}
}  // namespace kernel
