import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + div + gelu
fused_matmul_div_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_div_gelu_kernel(float* data, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Division
        float x = data[idx] / divisor;
        // GELU activation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        float tanh_inner = tanhf(inner);
        data[idx] = 0.5f * x * (1.0f + tanh_inner);
    }
}

torch::Tensor fused_matmul_div_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor) {
    // Perform matrix multiplication using cuBLAS
    auto output = torch::matmul(input, weight.transpose(0, 1)) + bias;
    
    // Launch kernel for element-wise division and GELU
    auto size = output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_div_gelu_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), divisor, size);
    
    return output;
}
"""

fused_matmul_div_gelu_cpp_source = (
    "torch::Tensor fused_matmul_div_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);"
)

# Compile the inline CUDA code for fused operation
fused_matmul_div_gelu = load_inline(
    name="fused_matmul_div_gelu",
    cpp_sources=fused_matmul_div_gelu_cpp_source,
    cuda_sources=fused_matmul_div_gelu_source,
    functions=["fused_matmul_div_gelu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused matmul + div + gelu operation.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.divisor = divisor
        self.fused_op = fused_matmul_div_gelu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.fused_op.fused_matmul_div_gelu_cuda(x, self.weight, self.bias, self.divisor)

batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, output_size, divisor]