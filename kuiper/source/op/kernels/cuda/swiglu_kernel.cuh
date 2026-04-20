#ifndef SWIGLU_KERNEL_CU_CUH
#define SWIGLU_KERNEL_CU_CUH
#include <tensor/tensor.h>
namespace kernel {
void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream);

void swiglu_w2_fused_kernel_cu(const tensor::Tensor& gate_tensor, const tensor::Tensor& up_tensor,
                               const tensor::Tensor& w2_weight, const tensor::Tensor& output,
                               void* stream);
}
#endif  // SWIGLU_KERNEL_CU_CUH
