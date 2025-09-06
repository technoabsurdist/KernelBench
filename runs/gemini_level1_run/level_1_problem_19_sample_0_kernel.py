import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for element-wise ReLU
// Computes out[i] = max(0, x[i]) for each element
__global__ void relu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(x[idx], 0.0f);
    }
}

// C++ wrapper function to launch the CUDA kernel from PyTorch
torch::Tensor relu_cuda(torch::Tensor x) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    // Create an output tensor with the same properties as the input
    auto out = torch::empty_like(x);
    auto size = x.numel();

    // Define kernel launch parameters
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    // Check for errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ function signature for the wrapper
relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor x);"

# Compile the inline CUDA code using torch.utils.cpp_extension.load_inline
# This creates a Python module that can be used to call the C++/CUDA function
relu_custom = load_inline(
    name="relu_custom",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a ReLU activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor using the custom kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        # Call the custom CUDA function from the compiled module
        return relu_custom.relu_cuda(x)