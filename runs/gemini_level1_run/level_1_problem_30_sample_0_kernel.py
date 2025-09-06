import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for a fused Softsign operation
softsign_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fabsf

__global__ void softsign_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Fused operation: y = x / (1 + |x|)
        out[idx] = x[idx] / (1.0f + fabsf(x[idx]));
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float tensor");

    // Allocate output tensor
    auto out = torch::empty_like(x);
    auto size = x.numel();

    // Avoid launching kernel for empty tensors
    if (size == 0) {
        return out;
    }

    // Kernel launch configuration
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    softsign_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    // Check for any CUDA errors that might have occurred during the kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# Define the C++ function signature
softsign_cpp_source = "torch::Tensor softsign_cuda(torch::Tensor x);"

# Compile the inline CUDA code
# This fuses the abs, add, and div operations into a single kernel launch,
# reducing memory bandwidth usage and kernel launch overhead.
fused_softsign = load_inline(
    name="fused_softsign",
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_cuda_source,
    functions=["softsign_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softsign activation using a custom fused CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softsign activation using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softsign applied, same shape as input.
        """
        return fused_softsign.softsign_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed