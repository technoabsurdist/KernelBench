import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + scale + residual
fused_matmul_scale_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void scale_and_residual_kernel(float* output, const float* original, float scaling_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = output[idx] * scaling_factor + original[idx];
    }
}

torch::Tensor fused_matmul_scale_residual_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor) {
    // Perform matrix multiplication: output = input @ weight.T + bias
    auto output = torch::mm(input, weight.transpose(0, 1));
    if (bias.defined()) {
        output = output + bias;
    }
    
    // Clone original output for residual connection
    auto original = output.clone();
    
    // Apply scaling and residual addition
    auto size = output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    scale_and_residual_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), 
        original.data_ptr<float>(), 
        scaling_factor, 
        size
    );
    
    return output;
}
"""

fused_matmul_scale_residual_cpp_source = """
torch::Tensor fused_matmul_scale_residual_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);
"""

# Compile the inline CUDA code for fused operation
fused_matmul_scale_residual = load_inline(
    name="fused_matmul_scale_residual",
    cpp_sources=fused_matmul_scale_residual_cpp_source,
    cuda_sources=fused_matmul_scale_residual_source,
    functions=["fused_matmul_scale_residual_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul + scale + residual.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_matmul_scale_residual

    def forward(self, x):
        """
        Forward pass using fused CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.fused_op.fused_matmul_scale_residual_cuda(
            x, self.weight, self.bias, self.scaling_factor
        )

batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]