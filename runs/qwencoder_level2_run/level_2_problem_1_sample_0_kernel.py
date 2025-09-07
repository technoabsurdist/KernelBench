import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv2d + relu + bias add
fused_conv_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void relu_bias_kernel(float* data, const float* bias, int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * height * width;
    
    if (idx < total_elements) {
        int channel_idx = (idx / (height * width)) % channels;
        float val = data[idx];
        val = fmaxf(0.0f, val);  // ReLU
        val += bias[channel_idx];  // Add bias
        data[idx] = val;
    }
}

torch::Tensor fused_conv_relu_bias_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                       int stride, int padding, int dilation) {
    // Perform convolution using PyTorch's built-in function
    auto conv_output = torch::conv2d(input, weight, {}, stride, padding, dilation, 1);
    
    // Get dimensions
    int batch = conv_output.size(0);
    int channels = conv_output.size(1);
    int height = conv_output.size(2);
    int width = conv_output.size(3);
    
    // Launch kernel for ReLU and bias addition
    int total_elements = batch * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    relu_bias_kernel<<<num_blocks, block_size>>>(
        conv_output.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        batch, channels, height, width
    );
    
    return conv_output;
}
"""

fused_conv_relu_bias_cpp_source = """
torch::Tensor fused_conv_relu_bias_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                       int stride, int padding, int dilation);
"""

# Compile the inline CUDA code for fused conv + relu + bias
fused_conv_relu_bias = load_inline(
    name="fused_conv_relu_bias",
    cpp_sources=fused_conv_relu_bias_cpp_source,
    cuda_sources=fused_conv_relu_bias_source,
    functions=["fused_conv_relu_bias_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused conv2d + relu + bias addition in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.fused_conv_relu_bias = fused_conv_relu_bias

    def forward(self, x):
        return self.fused_conv_relu_bias.fused_conv_relu_bias_cuda(x, self.weight, self.bias, 1, 1, 1)