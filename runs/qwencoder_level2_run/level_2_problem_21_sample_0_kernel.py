import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv+bias+scale+sigmoid
fused_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_conv_bias_scale_sigmoid_kernel(
    const float* conv_out,
    const float* bias,
    const float* scale,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int c = (idx / (height * width)) % channels;
        float val = conv_out[idx] + bias[c];
        val = val * scale[c];
        output[idx] = 1.0f / (1.0f + expf(-val)); // Sigmoid
    }
}

torch::Tensor fused_conv_bias_scale_sigmoid_cuda(
    torch::Tensor conv_out,
    torch::Tensor bias,
    torch::Tensor scale
) {
    auto batch_size = conv_out.size(0);
    auto channels = conv_out.size(1);
    auto height = conv_out.size(2);
    auto width = conv_out.size(3);
    
    auto output = torch::zeros_like(conv_out);
    
    const int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}
"""

fused_conv_cpp_source = """
torch::Tensor fused_conv_bias_scale_sigmoid_cuda(
    torch::Tensor conv_out,
    torch::Tensor bias,
    torch::Tensor scale
);
"""

# Compile the inline CUDA code for fused operation
fused_conv_op = load_inline(
    name="fused_conv_op",
    cpp_sources=fused_conv_cpp_source,
    cuda_sources=fused_conv_source,
    functions=["fused_conv_bias_scale_sigmoid_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels for conv+bias+scale+sigmoid operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_op = fused_conv_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_conv_bias_scale_sigmoid_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]