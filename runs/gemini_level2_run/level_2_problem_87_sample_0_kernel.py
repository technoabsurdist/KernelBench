import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused (subtract -> subtract -> mish)
fused_subtract_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For expf, log1pf, tanhf

// Mish activation: x * tanh(softplus(x))
// softplus(x) = log(1 + exp(x))
// Using log1pf(expf(x)) for better numerical stability than logf(1.0f + expf(x))
__device__ __forceinline__ float mish_activation(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_subtract_mish_kernel(const float* input, float* output, int size, float total_subtract) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Fuse the two subtractions and the Mish activation
        float val = input[idx] - total_subtract;
        output[idx] = mish_activation(val);
    }
}

torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float total_subtract) {
    // Input validation checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    // Allocate output tensor
    auto output = torch::empty_like(input);
    const int size = input.numel();

    // Avoid launching kernel for empty tensors
    if (size == 0) {
        return output;
    }

    // Kernel launch configuration
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    fused_subtract_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        total_subtract
    );

    // Check for any CUDA errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for the function signature, required by load_inline
fused_subtract_mish_cpp_source = """
torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float total_subtract);
"""

# JIT compile the CUDA kernel
fused_op = load_inline(
    name="fused_subtract_mish",
    cpp_sources=fused_subtract_mish_cpp_source,
    cuda_sources=fused_subtract_mish_source,
    functions=["fused_subtract_mish_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, followed by a custom CUDA kernel
    that fuses two subtractions and a Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Pre-compute the total subtraction value to pass to the kernel
        self.total_subtract = subtract_value_1 + subtract_value_2
        self.fused_op = fused_op

    def forward(self, x):
        # 1. Standard PyTorch convolution (highly optimized by cuDNN)
        x = self.conv(x)
        # 2. Custom fused kernel for element-wise operations
        x = self.fused_op.fused_subtract_mish_cuda(x, self.total_subtract)
        return x