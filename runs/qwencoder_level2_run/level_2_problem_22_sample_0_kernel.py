import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + scale + residual + clamp + logsumexp + mish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>

__global__ void fused_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size,
    float scale_factor,
    float clamp_min,
    float clamp_max
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
    
    // Compute matmul + bias for this output element
    float sum = 0.0f;
    for (int i = 0; i < input_size; ++i) {
        sum += input[batch_idx * input_size + i] * weight[hidden_idx * input_size + i];
    }
    sum += bias[hidden_idx];
    
    // Apply scale
    sum *= scale_factor;
    
    // Add residual (sum + sum)
    sum += sum;
    
    // Clamp
    sum = fmaxf(clamp_min, fminf(clamp_max, sum));
    
    // For simplicity in this kernel, we'll compute logsumexp and mish in a simplified way
    // In a production implementation, you'd want to do this more efficiently
    output[batch_idx * hidden_size + hidden_idx] = sum;
}

__global__ void logsumexp_mish_kernel(
    float* data,
    int batch_size,
    int hidden_size,
    float clamp_min,
    float clamp_max
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Compute max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < hidden_size; ++i) {
        float val = data[batch_idx * hidden_size + i];
        max_val = fmaxf(max_val, val);
    }
    
    // Compute sum of exp
    float sum = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        float val = data[batch_idx * hidden_size + i];
        sum += expf(val - max_val);
    }
    
    // Compute logsumexp
    float lse = logf(sum) + max_val;
    
    // Apply mish: x * tanh(softplus(x))
    for (int i = 0; i < hidden_size; ++i) {
        float x = data[batch_idx * hidden_size + i];
        float softplus = logf(1.0f + expf(x));
        float tanh_softplus = tanhf(softplus);
        data[batch_idx * hidden_size + i] = lse * (x * tanh_softplus);
    }
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max
) {
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch first kernel for matmul + scale + residual + clamp
    dim3 grid_dim(batch_size, (hidden_size + 255) / 256);
    dim3 block_dim(256);
    
    fused_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        scale_factor,
        clamp_min,
        clamp_max
    );
    
    // Launch second kernel for logsumexp + mish
    logsumexp_mish_kernel<<<batch_size, 1>>>(
        output.data_ptr<float>(),
        batch_size,
        hidden_size,
        clamp_min,
        clamp_max
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_operation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max
);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_operation = fused_operation

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        return self.fused_operation.fused_operation_cuda(
            x, self.weight, self.bias, 
            self.scale_factor, self.clamp_min, self.clamp_max
        )