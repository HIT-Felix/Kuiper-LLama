#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "base/profiler.h"
#include "swiglu_kernel.cuh"
namespace kernel {
__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();

  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;

  out[idx] = smem1[tid] * smem2[tid];
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK, int TILE_K>
__global__ void swiglu_w2_fused_kernel_fp32(const float* gate, const float* up,
                                            const float* w2_weight, float* output, int hidden_dim,
                                            int out_dim) {
  const unsigned int tid = threadIdx.x;
  const int start_row = blockIdx.x * ROW_PER_BLOCK;
  if (start_row >= out_dim) {
    return;
  }

  __shared__ float gate_tile[TILE_K];
  __shared__ float up_tile[TILE_K];
  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage reduce_storage[ROW_PER_BLOCK];

  float partial_sums[ROW_PER_BLOCK];
#pragma unroll
  for (int r = 0; r < ROW_PER_BLOCK; ++r) {
    partial_sums[r] = 0.f;
  }

  for (int tile_start = 0; tile_start < hidden_dim; tile_start += TILE_K) {
    const int tile_size = min(TILE_K, hidden_dim - tile_start);
    for (int i = tid; i < tile_size; i += THREAD_PER_BLOCK) {
      gate_tile[i] = gate[tile_start + i];
      up_tile[i] = up[tile_start + i];
    }
    __syncthreads();

#pragma unroll
    for (int r = 0; r < ROW_PER_BLOCK; ++r) {
      const int row = start_row + r;
      if (row >= out_dim) {
        continue;
      }

      const float* row_weight = w2_weight + row * hidden_dim + tile_start;
      float local_sum = 0.f;
      for (int i = tid; i < tile_size; i += THREAD_PER_BLOCK) {
        const float gate_val = gate_tile[i];
        const float gate_silu = gate_val / (1.0f + __expf(-gate_val));
        local_sum += gate_silu * up_tile[i] * row_weight[i];
      }
      partial_sums[r] += local_sum;
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < ROW_PER_BLOCK; ++r) {
    const int row = start_row + r;
    if (row >= out_dim) {
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

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  const size_t shmem = threads * sizeof(float) * 2;
  if (!stream) {
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  }
}

void swiglu_w2_fused_kernel_cu(const tensor::Tensor& gate_tensor, const tensor::Tensor& up_tensor,
                               const tensor::Tensor& w2_weight, const tensor::Tensor& output,
                               void* stream) {
  CHECK(!gate_tensor.is_empty());
  CHECK(!up_tensor.is_empty());
  CHECK(!w2_weight.is_empty());
  CHECK(!output.is_empty());
  CHECK_EQ(gate_tensor.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(up_tensor.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(w2_weight.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(output.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(gate_tensor.size(), up_tensor.size());
  CHECK_EQ(w2_weight.dims_size(), 2);
  CHECK_EQ(w2_weight.get_dim(1), static_cast<int32_t>(gate_tensor.size()));
  CHECK_EQ(w2_weight.get_dim(0), output.get_dim(0));

  constexpr int kThreadsPerBlock = 128;
  constexpr int kRowsPerBlock = 4;
  constexpr int kTileK = 256;
  const int hidden_dim = static_cast<int>(gate_tensor.size());
  const int out_dim = output.get_dim(0);
  const int block_num = (out_dim + kRowsPerBlock - 1) / kRowsPerBlock;

  base::ScopedCudaProfile profile("SwiGLUW2", stream ? static_cast<cudaStream_t>(stream) : nullptr);
  if (stream) {
    auto stream_ = static_cast<cudaStream_t>(stream);
    swiglu_w2_fused_kernel_fp32<kThreadsPerBlock, kRowsPerBlock, kTileK>
        <<<block_num, kThreadsPerBlock, 0, stream_>>>(gate_tensor.ptr<float>(), up_tensor.ptr<float>(),
                                                      w2_weight.ptr<float>(),
                                                      const_cast<float*>(output.ptr<float>()),
                                                      hidden_dim, out_dim);
  } else {
    swiglu_w2_fused_kernel_fp32<kThreadsPerBlock, kRowsPerBlock, kTileK>
        <<<block_num, kThreadsPerBlock>>>(gate_tensor.ptr<float>(), up_tensor.ptr<float>(),
                                          w2_weight.ptr<float>(),
                                          const_cast<float*>(output.ptr<float>()), hidden_dim,
                                          out_dim);
  }
}
}  // namespace kernel
