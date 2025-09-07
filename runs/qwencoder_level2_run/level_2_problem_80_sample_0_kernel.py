import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Max + Subtract Mean + GELU
fused_gemm_norm_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_norm_activation_kernel(
    const float* input,
    float* output,
    int batch_size,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx >= batch_size || feature_idx >= out_features) return;
    
    int idx = batch_idx * out_features + feature_idx;
    
    // Find max value for this batch
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = input[batch_idx * out_features];
        for (int i = 1; i < out_features; i++) {
            float val = input[batch_idx * out_features + i];
            shared_max = fmaxf(shared_max, val);
        }
    }
    __syncthreads();
    
    // Subtract max and compute mean
    float val = input[idx] - shared_max;
    __shared__ float shared_mean;
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < out_features; i++) {
            sum += input[batch_idx * out_features + i] - shared_max;
        }
        shared_mean = sum / out_features;
    }
    __syncthreads();
    
    // Subtract mean and apply GELU
    val -= shared_mean;
    
    // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float gelu_val = 0.5f * val * (1.0f + tanhf(sqrt_2_over_pi * (val + 0.044715f * val * val * val)));
    
    output[idx] = gelu_val;
}

torch::Tensor fused_gemm_norm_activation_cuda(
    torch::Tensor gemm_output
) {
    auto batch_size = gemm_output.size(0);
    auto out_features = gemm_output.size(1);
    
    auto output = torch::zeros_like(gemm_output);
    
    // Launch configuration
    const int threads_per_block = 256;
    const int blocks_per_row = (out_features + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, blocks_per_row);
    dim3 block(threads_per_block);
    
    fused_norm_activation_kernel<<<grid, block>>>(
        gemm_output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return output;
}
"""

fused_gemm_norm_activation_cpp_source = """
torch::Tensor fused_gemm_norm_activation_cuda(torch::Tensor gemm_output);
"""

# Compile the inline CUDA code for fused operation
fused_gemm_norm_activation = load_inline(
    name="fused_gemm_norm_activation",
    cpp_sources=fused_gemm_norm_activation_cpp_source,
    cuda_sources=fused_gemm_norm_activation_source,
    functions=["fused_gemm_norm_activation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused custom CUDA kernels.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.fused_op = fused_gemm_norm_activation

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        x = self.gemm(x)
        x = self.fused_op.fused_gemm_norm_activation_cuda(x)
        return x