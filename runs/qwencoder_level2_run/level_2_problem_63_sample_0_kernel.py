import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + relu + div
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_matmul_relu_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float divisor
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && out_idx < out_features) {
        float sum = 0.0f;
        
        // Compute matmul for this output element
        for (int i = 0; i < in_features; ++i) {
            sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }
        
        // Add bias
        sum += bias[out_idx];
        
        // Apply ReLU
        sum = fmaxf(sum, 0.0f);
        
        // Divide by constant
        sum = sum / divisor;
        
        // Store result
        output[batch_idx * out_features + out_idx] = sum;
    }
}

torch::Tensor fused_matmul_relu_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid_y = (out_features + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, blocks_per_grid_y);
    dim3 block(threads_per_block);
    
    fused_matmul_relu_div_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        divisor
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_relu_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_relu_div_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul + ReLU + div operation.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.divisor = divisor
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_matmul_relu_div_cuda(x, self.weight, self.bias, self.divisor)

batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, divisor]