import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activation functions and bias
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_impl(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

__global__ void fused_activation_kernel(
    const float* input, 
    const float* bias,
    float* output, 
    int batch_size,
    int channels,
    int spatial_size,
    float leaky_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        
        float val = input[idx];
        
        // ReLU
        val = fmaxf(0.0f, val);
        
        // LeakyReLU
        val = (val > 0) ? val : leaky_slope * val;
        
        // GELU
        val = gelu_impl(val);
        
        // Sigmoid
        val = 1.0f / (1.0f + expf(-val));
        
        // Add bias
        val = val + bias[c];
        
        output[idx] = val;
    }
}

torch::Tensor fused_activation_cuda(
    torch::Tensor input, 
    torch::Tensor bias,
    float leaky_slope
) {
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
    
    fused_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        leaky_slope
    );
    
    return output;
}
"""

fused_activation_cpp_source = """
torch::Tensor fused_activation_cuda(
    torch::Tensor input, 
    torch::Tensor bias,
    float leaky_slope
);
"""

# Compile the inline CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused activation functions and bias addition
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_activation_cuda(
            x, 
            self.bias.squeeze(-1).squeeze(-1).squeeze(-1),
            0.01
        )
        return x

batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]