import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + min + subtract
fused_linear_min_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_linear_min_subtract_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float constant_val,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && out_idx < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }
        sum += bias[out_idx];
        
        // Apply min and subtract
        float result = fminf(sum, constant_val) - constant_val;
        output[batch_idx * out_features + out_idx] = result;
    }
}

torch::Tensor fused_linear_min_subtract_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float constant_val
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 grid(batch_size, (out_features + 255) / 256);
    dim3 block(256);
    
    fused_linear_min_subtract_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        constant_val,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_linear_min_subtract_cpp_source = """
torch::Tensor fused_linear_min_subtract_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float constant_val
);
"""

# Compile the inline CUDA code for fused operation
fused_op = load_inline(
    name="fused_linear_min_subtract",
    cpp_sources=fused_linear_min_subtract_cpp_source,
    cuda_sources=fused_linear_min_subtract_source,
    functions=["fused_linear_min_subtract_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for linear + min + subtract operations.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.constant = constant
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_linear_min_subtract_cuda(x, self.weight, self.bias, self.constant)

batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, constant]