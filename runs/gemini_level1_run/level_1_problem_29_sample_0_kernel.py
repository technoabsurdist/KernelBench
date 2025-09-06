import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for a numerically stable Softplus
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softplus_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        // For large positive x, softplus(x) is approximately x.
        // This check avoids overflow in expf(x) and maintains precision.
        if (x_val > 20.0f) {
            out[idx] = x_val;
        } else {
            out[idx] = logf(1.0f + expf(x_val));
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    auto out = torch::empty_like(x);
    int size = x.numel();

    if (size == 0) {
        return out;
    }

    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    // Check for errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ interface definition for the CUDA function
softplus_cpp_source = """
torch::Tensor softplus_cuda(torch::Tensor x);
"""

# JIT compile the custom CUDA operator
softplus_custom_op = load_inline(
    name="softplus_custom_op",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs a Softplus activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus_op = softplus_custom_op
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using the custom kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return self.softplus_op.softplus_cuda(x)