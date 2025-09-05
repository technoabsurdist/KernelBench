import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused multiplication and LeakyReLU
fused_mul_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_mul_leaky_relu_kernel(
    float* x, 
    const float multiplier, 
    const float negative_slope, 
    const int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] * multiplier;
        x[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor fused_mul_leaky_relu_cuda(
    torch::Tensor x, 
    float multiplier, 
    float negative_slope) 
{
    auto size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_mul_leaky_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        multiplier, 
        negative_slope, 
        size
    );

    return x;
}
"""

fused_mul_leaky_relu_cpp_source = """
torch::Tensor fused_mul_leaky_relu_cuda(torch::Tensor x, float multiplier, float negative_slope);
"""

# Compile the inline CUDA code
fused_mul_leaky_relu = load_inline(
    name="fused_mul_leaky_relu",
    cpp_sources=fused_mul_leaky_relu_cpp_source,
    cuda_sources=fused_mul_leaky_relu_source,
    functions=["fused_mul_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused multiplication and LeakyReLU operations.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.fused_mul_leaky_relu = fused_mul_leaky_relu

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_mul_leaky_relu.fused_mul_leaky_relu_cuda(
            x.contiguous(), 
            self.multiplier, 
            self.negative_slope
        )
        return x

batch_size = 1024
in_features  = 8192  
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]