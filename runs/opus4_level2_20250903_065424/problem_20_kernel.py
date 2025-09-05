import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias add and residual operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_residual_kernel(
    float* x, 
    const float* bias,
    const float* original_x,
    int batch_size,
    int channels, 
    int depth,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * depth * height * width;
    
    if (idx < total_size) {
        int c = (idx / (depth * height * width)) % channels;
        
        float orig_val = original_x[idx];
        float x_val = x[idx];
        
        // Fused operations: x = x + bias, x = x + original_x, x = x * original_x, x = x + original_x
        // This simplifies to: x = original_x * (2 * original_x + bias + 1)
        x[idx] = orig_val * (2.0f * orig_val + bias[c] + 1.0f);
    }
}

torch::Tensor fused_bias_residual_cuda(torch::Tensor x, torch::Tensor bias) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);
    
    auto original_x = x.clone();
    
    int total_size = batch_size * channels * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_bias_residual_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        original_x.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width
    );
    
    return x;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_bias_residual_cuda(torch::Tensor x, torch::Tensor bias);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_bias_residual_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for post-convolution operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused kernel handles: bias add, residual adds, and multiplication
        x = self.fused_ops.fused_bias_residual_cuda(x.contiguous(), self.bias.squeeze())
        return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]