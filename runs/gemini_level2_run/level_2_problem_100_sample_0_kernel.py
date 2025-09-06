import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused clamp and division
fused_clamp_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf

__global__ void fused_clamp_div_kernel(const float* input, float* output, int size, float min_value, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Fused operation: clamp(min) followed by division
        float val = input[idx];
        output[idx] = fmaxf(val, min_value) / divisor;
    }
}

torch::Tensor fused_clamp_div_cuda(torch::Tensor input, float min_value, float divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto out = torch::empty_like(input);
    auto size = input.numel();

    if (size == 0) {
        return out;
    }

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_clamp_div_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        size,
        min_value,
        divisor
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_clamp_div_cpp_source = (
    "torch::Tensor fused_clamp_div_cuda(torch::Tensor input, float min_value, float divisor);"
)

# Compile the inline CUDA code for the fused operation
fused_clamp_div = load_inline(
    name="fused_clamp_div",
    cpp_sources=fused_clamp_div_cpp_source,
    cuda_sources=fused_clamp_div_source,
    functions=["fused_clamp_div_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, followed by a fused
    clamp and division operation using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        # Keep the highly optimized ConvTranspose3d from PyTorch
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # Store constants for the custom kernel
        self.min_value = min_value
        self.divisor = divisor
        
        # Store the compiled custom function
        self.fused_op = fused_clamp_div.fused_clamp_div_cuda

    def forward(self, x):
        # Step 1: Perform the convolution using the standard PyTorch operator
        x = self.conv_transpose(x)
        
        # Step 2: Apply the fused clamp and division using the custom CUDA kernel
        x = self.fused_op(x, self.min_value, self.divisor)
        
        return x