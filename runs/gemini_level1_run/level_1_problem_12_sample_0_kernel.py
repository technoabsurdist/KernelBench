import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the diag(A) @ B operation.
# This kernel avoids the explicit creation of the large diagonal matrix.
# The operation C = diag(A) @ B simplifies to an element-wise multiplication
# of each row of B by the corresponding element of A: C[i, j] = A[i] * B[i, j].
diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* a, const float* b, float* out, int N, int M) {
    // Calculate the global row and column indices using a 2D grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to ensure we don't write outside the output tensor
    if (row < N && col < M) {
        // Calculate the linear index for the 2D tensors B and out (row-major)
        int idx = row * M + col;
        // Perform the simplified operation: out[row, col] = a[row] * b[row, col]
        out[idx] = a[row] * b[idx];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    // Input validation
    TORCH_CHECK(a.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Input tensor B must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "Input tensor A must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input tensor B must be contiguous");
    TORCH_CHECK(a.dim() == 1, "Input tensor A must be 1D");
    TORCH_CHECK(b.dim() == 2, "Input tensor B must be 2D");
    TORCH_CHECK(a.size(0) == b.size(0), "Dimension mismatch: a.size(0) must equal b.size(0)");

    // Get dimensions
    const int N = b.size(0);
    const int M = b.size(1);

    // Create the output tensor with the same shape as B
    auto out = torch::empty_like(b);

    // Define CUDA launch configuration
    const dim3 block_size(16, 16);
    const dim3 num_blocks(
        (M + block_size.x - 1) / block_size.x,
        (N + block_size.y - 1) / block_size.y
    );

    // Launch the kernel
    diag_matmul_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        M
    );

    return out;
}
"""

diag_matmul_cpp_source = (
    "torch::Tensor diag_matmul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code
diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel to perform the operation C = diag(A) * B.
    This avoids materializing the large, sparse diagonal matrix, leading to significant
    memory and computation savings.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul_op = diag_matmul.diag_matmul_cuda

    def forward(self, A, B):
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return self.diag_matmul_op(A, B)