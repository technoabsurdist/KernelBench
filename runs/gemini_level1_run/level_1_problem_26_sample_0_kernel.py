import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For erff

// A constant for 1/sqrt(2)
#define M_SQRT1_2 0.7071067811865475f

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        out[idx] = 0.5f * val * (1.0f + erff(val * M_SQRT1_2));
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    // Ensure the input tensor is a contiguous CUDA tensor of floats
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    auto out = torch::empty_like(x);
    auto size = x.numel();

    if (size == 0) {
        return out;
    }

    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ source for the function signature, used by load_inline
gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for GELU
gelu_module = load_inline(
    name="gelu_module",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a GELU activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled CUDA function
        self.gelu_cuda = gelu_module.gelu_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor using the custom kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return self.gelu_cuda(x)