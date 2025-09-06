import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For expf

// Standard SELU constants
const float ALPHA = 1.6732632423543772f;
const float SCALE = 1.0507009873554805f;

__global__ void selu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val > 0.0f) {
            out[idx] = SCALE * val;
        } else {
            out[idx] = SCALE * (ALPHA * (expf(val) - 1.0f));
        }
    }
}

torch::Tensor selu_cuda(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be a float32 tensor");

    // Ensure tensor is contiguous in memory for kernel access
    auto x_contiguous = x.contiguous();

    auto out = torch::empty_like(x_contiguous);
    auto size = x_contiguous.numel();

    if (size == 0) {
        return out;
    }

    // Kernel launch configuration
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    selu_kernel<<<num_blocks, block_size>>>(
        x_contiguous.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );

    // Check for errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature, required by load_inline
selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor x);"

# Compile the inline CUDA code
# This creates a Python module with the 'selu_cuda' function
selu_op = load_inline(
    name="selu_op",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs a SELU activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled CUDA operator
        self.selu_op = selu_op
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor using the custom kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return self.selu_op.selu_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed