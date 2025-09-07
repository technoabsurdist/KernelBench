import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + subtract + multiply + relu
fused_linear_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_linear_sub_mul_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float subtract_value,
    float multiply_value
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && out_idx < out_features) {
        float sum = 0.0f;
        
        // Compute linear transformation
        for (int i = 0; i < in_features; ++i) {
            sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }
        
        // Add bias
        sum += bias[out_idx];
        
        // Subtract, multiply, and apply ReLU
        sum = (sum - subtract_value) * multiply_value;
        sum = fmaxf(0.0f, sum);
        
        output[batch_idx * out_features + out_idx] = sum;
    }
}

torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    const int threads_per_block = 256;
    const int blocks_per_grid_y = (out_features + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, blocks_per_grid_y);
    dim3 block(threads_per_block);
    
    fused_linear_sub_mul_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        subtract_value,
        multiply_value
    );
    
    return output;
}
"""

fused_linear_activation_cpp_source = """
torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value
);
"""

# Compile the inline CUDA code for fused operation
fused_linear_activation = load_inline(
    name="fused_linear_activation",
    cpp_sources=fused_linear_activation_cpp_source,
    cuda_sources=fused_linear_activation_source,
    functions=["fused_linear_sub_mul_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for linear + subtract + multiply + ReLU
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        
        # Initialize linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Load the custom CUDA function
        self.fused_op = fused_linear_activation

    def forward(self, x):
        return self.fused_op.fused_linear_sub_mul_relu_cuda(
            x, self.weight, self.bias, self.subtract_value, self.multiply_value
        )

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]