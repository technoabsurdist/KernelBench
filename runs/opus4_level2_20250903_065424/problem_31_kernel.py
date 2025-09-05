import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min + add + multiply operations
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(
    float* x, 
    const float* bias, 
    const float constant_value,
    const float scaling_factor,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;
    
    if (idx < total_size) {
        int c = (idx / (height * width)) % channels;
        
        // Apply min with constant
        float val = fminf(x[idx], constant_value);
        
        // Add bias (bias is shape [channels, 1, 1])
        val = val + bias[c];
        
        // Multiply by scaling factor
        x[idx] = val * scaling_factor;
    }
}

torch::Tensor fused_post_conv_cuda(
    torch::Tensor x, 
    torch::Tensor bias, 
    float constant_value,
    float scaling_factor) {
    
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    
    auto x_out = x.clone();
    auto total_size = batch_size * channels * height * width;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_post_conv_kernel<<<num_blocks, block_size>>>(
        x_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        constant_value,
        scaling_factor,
        batch_size,
        channels,
        height,
        width
    );
    
    return x_out;
}
"""

fused_post_conv_cpp_source = """
torch::Tensor fused_post_conv_cuda(
    torch::Tensor x, 
    torch::Tensor bias, 
    float constant_value,
    float scaling_factor);
"""

# Compile the inline CUDA code
fused_post_conv = load_inline(
    name="fused_post_conv",
    cpp_sources=fused_post_conv_cpp_source,
    cuda_sources=fused_post_conv_source,
    functions=["fused_post_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused post-convolution operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_post_conv = fused_post_conv

    def forward(self, x):
        x = self.conv(x)
        # Fused min + add + multiply operations
        x = self.fused_post_conv.fused_post_conv_cuda(
            x.contiguous(), 
            self.bias, 
            self.constant_value, 
            self.scaling_factor
        )
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]