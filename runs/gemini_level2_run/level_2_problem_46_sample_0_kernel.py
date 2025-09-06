import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: sub -> tanh -> sub
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For tanhf

__global__ void fused_op_kernel(const float* in, float* out, float val1, float val2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Fused operation: y = tanh(x - val1) - val2
        out[idx] = tanhf(in[idx] - val1) - val2;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor x, float val1, float val2) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    // Allocate output tensor
    auto out = torch::empty_like(x);
    auto size = x.numel();

    // Handle empty tensor case
    if (size == 0) {
        return out;
    }

    // Kernel launch configuration
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    fused_op_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        val1,
        val2,
        size
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in fused_op_kernel: ", cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ source for the function signature
fused_op_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor x, float val1, float val2);
"""

# Compile the inline CUDA code for the fused operation
# This fuses (x - subtract1_value), torch.tanh(x), and (x - subtract2_value)
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, a fused (subtraction-tanh-subtraction) operation, and average pooling.
    The element-wise operations are fused into a single custom CUDA kernel for performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        # Use standard PyTorch operators for complex, highly-optimized layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        
        # Store values needed for the custom kernel
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        
        # Assign the compiled custom operator
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        # Replace the sequence of three element-wise ops with a single call to our fused kernel
        x = self.fused_op.fused_op_cuda(x, self.subtract1_value, self.subtract2_value)
        x = self.avgpool(x)
        return x