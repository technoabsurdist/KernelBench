import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_scale_tanh_bias_sigmoid_kernel(
    const float* input,
    const float* scaling_factor, 
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int spatial_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        
        float val = input[idx];
        val = val * scaling_factor[c];
        val = tanhf(val);
        val = val * bias[c];
        val = 1.0f / (1.0f + expf(-val));
        output[idx] = val;
    }
}

torch::Tensor fused_scale_tanh_bias_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor scaling_factor,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto spatial_size = depth * height * width;
    
    auto output = torch::empty_like(input);
    
    int total_size = batch_size * channels * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_scale_tanh_bias_sigmoid_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scaling_factor.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_scale_tanh_bias_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor scaling_factor,
    torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_scale_tanh_bias_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for element-wise operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # Reshape scaling_factor and bias for the kernel
        scaling_factor = self.scaling_factor.view(-1)
        bias = self.bias.view(-1)
        x = self.fused_ops.fused_scale_tanh_bias_sigmoid_cuda(x, scaling_factor, bias)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]