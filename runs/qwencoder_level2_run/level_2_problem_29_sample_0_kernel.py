import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + double mish
fused_matmul_double_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_matmul_double_mish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
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
        
        // Apply Mish twice
        float result = mish(mish(sum));
        output[batch_idx * out_features + out_idx] = result;
    }
}

torch::Tensor fused_matmul_double_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    dim3 grid(batch_size, (out_features + block_size - 1) / block_size);
    dim3 block(block_size);
    
    fused_matmul_double_mish_kernel<<<grid, block>>>(
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

fused_matmul_double_mish_cpp_source = """
torch::Tensor fused_matmul_double_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for fused matmul + double mish
fused_matmul_double_mish = load_inline(
    name="fused_matmul_double_mish",
    cpp_sources=fused_matmul_double_mish_cpp_source,
    cuda_sources=fused_matmul_double_mish_source,
    functions=["fused_matmul_double_mish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused matmul + double mish CUDA kernel
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_op = fused_matmul_double_mish

    def forward(self, x):
        return self.fused_op.fused_matmul_double_mish_cuda(x, self.weight, self.bias)

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]