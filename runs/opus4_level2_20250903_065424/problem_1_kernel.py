import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ReLU + bias addition
fused_relu_add_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_add_bias_kernel(float* x, const float* bias, int batch_size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        float val = x[idx];
        val = fmaxf(0.0f, val);  // ReLU
        x[idx] = val + bias[c];   // Add bias
    }
}

torch::Tensor fused_relu_add_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto spatial_size = height * width;
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels * spatial_size + block_size - 1) / block_size;
    
    fused_relu_add_bias_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        batch_size, 
        channels, 
        spatial_size
    );
    
    return x;
}
"""

fused_relu_add_bias_cpp_source = (
    "torch::Tensor fused_relu_add_bias_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused ReLU + bias addition
fused_relu_add_bias = load_inline(
    name="fused_relu_add_bias",
    cpp_sources=fused_relu_add_bias_cpp_source,
    cuda_sources=fused_relu_add_bias_source,
    functions=["fused_relu_add_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused ReLU + bias addition kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_relu_add_bias = fused_relu_add_bias

    def forward(self, x):
        x = self.conv(x)
        # Fused ReLU + bias addition
        x = self.fused_relu_add_bias.fused_relu_add_bias_cuda(x.contiguous(), self.bias.squeeze())
        return x


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]