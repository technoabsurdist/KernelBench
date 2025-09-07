import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Sigmoid + Scaling + Residual Add
fused_gemm_sigmoid_scaling_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_sigmoid_scale_residual_kernel(const float* input, const float* weight, const float* bias, 
                                                    float* output, int batch_size, int input_size, int hidden_size, 
                                                    float scaling_factor) {
    int batch_idx = blockIdx.x;
    int hidden_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && hidden_idx < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[batch_idx * input_size + i] * weight[hidden_idx * input_size + i];
        }
        sum += bias[hidden_idx];
        
        // Apply sigmoid
        float sigmoid_val = 1.0f / (1.0f + expf(-sum));
        
        // Scale and add residual
        output[batch_idx * hidden_size + hidden_idx] = sigmoid_val * scaling_factor + sum;
    }
}

torch::Tensor fused_gemm_sigmoid_scaling_residual_cuda(torch::Tensor input, torch::Tensor weight, 
                                                       torch::Tensor bias, float scaling_factor) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 block_size(256);
    dim3 num_blocks(batch_size, (hidden_size + block_size.x - 1) / block_size.x);
    
    fused_sigmoid_scale_residual_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, input_size, hidden_size, scaling_factor);
        
    return output;
}
"""

fused_gemm_sigmoid_scaling_residual_cpp_source = """
torch::Tensor fused_gemm_sigmoid_scaling_residual_cuda(torch::Tensor input, torch::Tensor weight, 
                                                       torch::Tensor bias, float scaling_factor);
"""

# Compile the inline CUDA code
fused_gemm_sigmoid_scaling_residual = load_inline(
    name="fused_gemm_sigmoid_scaling_residual",
    cpp_sources=fused_gemm_sigmoid_scaling_residual_cpp_source,
    cuda_sources=fused_gemm_sigmoid_scaling_residual_source,
    functions=["fused_gemm_sigmoid_scaling_residual_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + Sigmoid + Scaling + Residual Add operation.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_gemm_sigmoid_scaling_residual

    def forward(self, x):
        """
        Forward pass of the optimized model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        return self.fused_op.fused_gemm_sigmoid_scaling_residual_cuda(
            x, self.weight, self.bias, self.scaling_factor)