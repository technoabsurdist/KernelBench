import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Bias + ReLU
fused_gemm_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_gemm_bias_relu_kernel(const float* __restrict__ input,
                                           const float* __restrict__ weight,
                                           const float* __restrict__ bias,
                                           float* __restrict__ output,
                                           const int batch_size,
                                           const int in_features,
                                           const int out_features) {
    const int batch_idx = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && out_idx < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }
        
        // Add bias and apply ReLU
        float result = sum + bias[out_idx];
        if (result < 0.0f) result = 0.0f;
        
        output[batch_idx * out_features + out_idx] = result;
    }
}

torch::Tensor fused_gemm_bias_relu_cuda(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, torch::kCUDA);
    
    const int threads_per_block = 256;
    const dim3 block_dim(1, (out_features + threads_per_block - 1) / threads_per_block, 1);
    const dim3 thread_dim(threads_per_block, 1, 1);
    
    fused_gemm_bias_relu_kernel<<<block_dim, thread_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_gemm_bias_relu_cpp_source = """
torch::Tensor fused_gemm_bias_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for fused GEMM + Bias + ReLU
fused_gemm_bias_relu = load_inline(
    name="fused_gemm_bias_relu",
    cpp_sources=fused_gemm_bias_relu_cpp_source,
    cuda_sources=fused_gemm_bias_relu_source,
    functions=["fused_gemm_bias_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + Bias + ReLU operation using custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_gemm_bias_relu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        return self.fused_op.fused_gemm_bias_relu_cuda(x, self.weight, self.bias)

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bias_shape]