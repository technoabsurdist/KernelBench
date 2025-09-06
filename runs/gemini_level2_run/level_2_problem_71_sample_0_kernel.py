import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused division and LeakyReLU
fused_div_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_div_leaky_relu_kernel(const float* input, float* output, int size, float divisor, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Perform division
        float val = input[idx] / divisor;
        // Apply LeakyReLU
        output[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor fused_div_leaky_relu_cuda(torch::Tensor input, float divisor, float negative_slope) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);
    const int size = input.numel();

    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_div_leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        divisor,
        negative_slope
    );
    
    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

fused_div_leaky_relu_cpp_source = (
    "torch::Tensor fused_div_leaky_relu_cuda(torch::Tensor input, float divisor, float negative_slope);"
)

# JIT compile the CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_div_leaky_relu_cpp_source,
    cuda_sources=fused_div_leaky_relu_source,
    functions=["fused_div_leaky_relu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the division and LeakyReLU operations into a single custom CUDA kernel.
    The convolution operation remains the highly optimized PyTorch implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.negative_slope = 0.01
        # Store the compiled CUDA function
        self.fused_div_leaky_relu = fused_op.fused_div_leaky_relu_cuda

    def forward(self, x):
        # 1. Use the standard, highly optimized Conv2d implementation
        x = self.conv(x)
        # 2. Apply the custom fused kernel for division and LeakyReLU
        x = self.fused_div_leaky_relu(x, self.divisor, self.negative_slope)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]