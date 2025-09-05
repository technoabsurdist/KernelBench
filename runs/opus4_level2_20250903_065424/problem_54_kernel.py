import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused multiply + LeakyReLU + GELU
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float gelu_fwd(float x) {
    const float c = 0.797884560803f; // sqrt(2/pi)
    const float a = 0.044715f;
    
    float x3 = x * x * x;
    float inner = c * (x + a * x3);
    float tanh_inner = tanhf(inner);
    
    return 0.5f * x * (1.0f + tanh_inner);
}

__global__ void fused_multiply_leakyrelu_gelu_kernel(
    const float* input,
    const float* multiplier,
    float* output,
    int total_size,
    int spatial_size,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        int channel_idx = (idx / spatial_size) % gridDim.y;
        
        float val = input[idx] * multiplier[channel_idx];
        
        // LeakyReLU
        val = val > 0 ? val : negative_slope * val;
        
        // GELU
        val = gelu_fwd(val);
        
        output[idx] = val;
    }
}

torch::Tensor fused_multiply_leakyrelu_gelu_cuda(
    torch::Tensor input,
    torch::Tensor multiplier,
    float negative_slope
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::empty_like(input);
    
    int total_size = batch_size * channels * height * width;
    int spatial_size = height * width;
    
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    
    dim3 grid(blocks, channels);
    
    fused_multiply_leakyrelu_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        total_size,
        spatial_size,
        negative_slope
    );
    
    return output;
}
"""

fused_activation_cpp_source = """
torch::Tensor fused_multiply_leakyrelu_gelu_cuda(
    torch::Tensor input,
    torch::Tensor multiplier,
    float negative_slope
);
"""

# Compile the inline CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_multiply_leakyrelu_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused multiply + LeakyReLU + GELU kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.negative_slope = 0.01  # Default LeakyReLU negative slope
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        # Reshape multiplier to be 1D for the kernel
        multiplier_flat = self.multiplier.view(-1)
        x = self.fused_activation.fused_multiply_leakyrelu_gelu_cuda(
            x, multiplier_flat, self.negative_slope
        )
        return x

batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]