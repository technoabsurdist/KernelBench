import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Mish and Tanh activations
fused_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For tanhf, log1pf, expf

// Device function for a numerically stable softplus: log(1 + exp(x))
__device__ __forceinline__ float softplus_dev(float x) {
    // Use a threshold to avoid overflow in expf(x) for large x
    // and to maintain precision. For x > 20, softplus(x) is very close to x.
    if (x > 20.0f) {
        return x;
    } else {
        return log1pf(expf(x));
    }
}

// Device function for Mish activation: x * tanh(softplus(x))
__device__ __forceinline__ float mish_dev(float x) {
    return x * tanhf(softplus_dev(x));
}

// Fused kernel that computes tanh(mish(x)) for each element
__global__ void fused_mish_tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Fused operation: tanh(mish(val))
        output[idx] = tanhf(mish_dev(val));
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_mish_tanh_cuda(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    // Allocate output tensor
    auto out = torch::empty_like(input);
    auto size = input.numel();

    // Avoid launching kernel for empty tensors
    if (size == 0) {
        return out;
    }

    // Kernel launch configuration
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    fused_mish_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );

    // Check for any CUDA errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# Define the C++ source for the function signature
fused_mish_tanh_cpp_source = """
torch::Tensor fused_mish_tanh_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
fused_mish_tanh = load_inline(
    name="fused_mish_tanh",
    cpp_sources=fused_mish_tanh_cpp_source,
    cuda_sources=fused_mish_tanh_source,
    functions=["fused_mish_tanh_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, then applies a fused Mish+Tanh activation
    using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        # The Conv3d layer remains a standard PyTorch operator as it's highly optimized.
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Store the compiled custom operator
        self.fused_activations = fused_mish_tanh

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        # 1. Apply the standard, highly optimized Conv3d layer
        x = self.conv(x)
        # 2. Apply the custom fused activation kernel, replacing two separate operations
        x = self.fused_activations.fused_mish_tanh_cuda(x)
        return x