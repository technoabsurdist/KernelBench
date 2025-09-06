import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and its C++ wrapper for matrix-scalar multiplication
matrix_scalar_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel to perform element-wise multiplication of a matrix by a scalar
__global__ void matrix_scalar_mul_kernel(const float* matrix, float scalar, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = matrix[idx] * scalar;
    }
}

// C++ wrapper function that launches the CUDA kernel
torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s) {
    // Ensure the input tensor is a contiguous CUDA tensor
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "Input tensor A must be contiguous");

    // Get the total number of elements in the tensor
    const int num_elements = A.numel();
    
    // Create an output tensor with the same shape and device as the input
    auto C = torch::empty_like(A);

    // Configure the grid and block dimensions for the kernel launch
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    // Launch the kernel
    matrix_scalar_mul_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(),
        s,
        C.data_ptr<float>(),
        num_elements
    );
    
    return C;
}
"""

# C++ source for the function declaration (needed for linking)
matrix_scalar_mul_cpp_source = """
torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s);
"""

# JIT compile the CUDA and C++ code into a loadable module
matrix_scalar_mul_op = load_inline(
    name="matrix_scalar_mul_op",
    cpp_sources=matrix_scalar_mul_cpp_source,
    cuda_sources=matrix_scalar_mul_source,
    functions=["matrix_scalar_mul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for matrix-scalar multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using the custom CUDA kernel.

        Args:
            A: Input matrix of shape (M, N) on the CUDA device.
            s: Scalar value.

        Returns:
            C: Resulting matrix of shape (M, N).
        """
        return matrix_scalar_mul_op.matrix_scalar_mul_cuda(A, s)