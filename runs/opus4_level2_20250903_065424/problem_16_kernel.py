import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Mish + Add + Hardtanh + Scale
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float mish(float x) {
    return x * tanh(log1p(exp(x)));
}

__global__ void fused_activation_kernel(float* x, const float add_value, const float scale, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Apply Mish activation
        val = mish(val);
        // Add value
        val = val + add_value;
        // Apply Hardtanh (clamp between -1 and 1)
        val = fminf(fmaxf(val, -1.0f), 1.0f);
        // Scale
        val = val * scale;
        x[idx] = val;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor x, float add_value, float scale) {
    auto size = x.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_activation_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        add_value, 
        scale, 
        size
    );
    
    return x;
}
"""

fused_activation_cpp_source = """
torch::Tensor fused_activation_cuda(torch::Tensor x, float add_value, float scale);
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
    Optimized model with fused CUDA kernel for Mish + Add + Hardtanh + Scale operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused Mish + Add + Hardtanh + Scale in a single kernel
        x = self.fused_activation.fused_activation_cuda(x.contiguous(), self.add_value, self.scale)
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128  
kernel_size  = 3
stride       = 2  
padding      = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]