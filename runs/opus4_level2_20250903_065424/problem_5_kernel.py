import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias subtraction and tanh
bias_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void bias_subtract_tanh_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;
    
    if (idx < total_size) {
        int c = (idx / (height * width)) % channels;
        float val = input[idx] - bias[c];
        output[idx] = tanhf(val);
    }
}

torch::Tensor bias_subtract_tanh_cuda(torch::Tensor input, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    int total_size = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    bias_subtract_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}
"""

bias_tanh_cpp_source = "torch::Tensor bias_subtract_tanh_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the inline CUDA code
bias_tanh = load_inline(
    name="bias_tanh",
    cpp_sources=bias_tanh_cpp_source,
    cuda_sources=bias_tanh_source,
    functions=["bias_subtract_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused bias subtraction and tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.bias_tanh = bias_tanh

    def forward(self, x):
        x = self.conv_transpose(x)
        # Reshape bias for the custom kernel
        bias_flat = self.bias.view(-1)
        x = self.bias_tanh.bias_subtract_tanh_cuda(x, bias_flat)
        return x

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]