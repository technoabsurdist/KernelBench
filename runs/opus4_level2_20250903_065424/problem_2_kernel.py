import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-processing operations
fused_postprocess_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_postprocess_kernel(
    float* x, 
    const float* bias, 
    float scaling_factor,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int c = (idx / (height * width)) % channels;
        
        // Add bias
        float val = x[idx] + bias[c];
        
        // First clamp [0, 1]
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        
        // Scale
        val = val * scaling_factor;
        
        // Second clamp [0, 1]
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        
        // Divide by scaling factor
        val = val / scaling_factor;
        
        x[idx] = val;
    }
}

torch::Tensor fused_postprocess_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    float scaling_factor) {
    
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    
    int total_elements = batch_size * channels * height * width;
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_postprocess_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        batch_size,
        channels,
        height,
        width
    );
    
    return x;
}
"""

fused_postprocess_cpp_source = """
torch::Tensor fused_postprocess_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    float scaling_factor);
"""

# Compile the inline CUDA code
fused_postprocess = load_inline(
    name="fused_postprocess",
    cpp_sources=fused_postprocess_cpp_source,
    cuda_sources=fused_postprocess_source,
    functions=["fused_postprocess_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for post-processing operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape).squeeze(-1).squeeze(-1))  # Shape: (out_channels,)
        self.scaling_factor = scaling_factor
        self.fused_postprocess = fused_postprocess

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_postprocess.fused_postprocess_cuda(
            x.contiguous(), 
            self.bias, 
            self.scaling_factor
        )
        return x


batch_size = 128
in_channels = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]