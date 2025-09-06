import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused lower-triangular matrix multiplication
tril_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to compute C = A * B where A and B are lower triangular matrices.
// The result C is also a lower triangular matrix.
__global__ void tril_matmul_kernel(const float* A, const float* B, float* C, int M) {
    // Each thread computes one element of the output matrix C.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (row < M && col < M) {
        // The result of multiplying two lower triangular matrices is another
        // lower triangular matrix. We only need to compute C[row, col]
        // where col <= row. Threads responsible for the upper triangle (col > row)
        // can exit early, saving computation.
        if (col > row) {
            // The output tensor C is pre-initialized to zeros, so we don't need
            // to explicitly write zero here.
            return;
        }

        float C_value = 0.0f;

        // Standard matrix multiplication is C[row, col] = sum_{k} A[row, k] * B[k, col].
        // We can optimize the summation range based on the triangular property.
        // - A is lower triangular, so A[row, k] is 0 for k > row.
        // - B is lower triangular, so B[k, col] is 0 for col > k.
        // Combining these, the sum over k only needs to go from k=col to k=row.
        // This significantly reduces the number of multiplications.
        for (int k = col; k <= row; ++k) {
            C_value += A[row * M + k] * B[k * M + col];
        }

        C[row * M + col] = C_value;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor tril_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    const int M = A.size(0);

    // Create an output tensor initialized to zeros. Our kernel will only fill
    // the lower triangular part, leaving the upper part as zero.
    auto C = torch::zeros_like(A);

    // Define CUDA grid and block dimensions
    const int TILE_DIM = 16;
    const dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    const dim3 numBlocks(
        (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch the kernel
    tril_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M
    );

    return C;
}
"""

# C++ source for the function signature, required by load_inline
tril_matmul_cpp_source = """
torch::Tensor tril_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# JIT compile the CUDA and C++ code
tril_matmul = load_inline(
    name="tril_matmul",
    cpp_sources=tril_matmul_cpp_source,
    cuda_sources=tril_matmul_source,
    functions=["tril_matmul_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel to perform matrix multiplication
    of two lower triangular matrices. This fuses the matmul and tril operations
    and avoids redundant computations on zero-elements.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # The compiled kernel is now available as a Python function
        self.tril_matmul_op = tril_matmul.tril_matmul_cuda

    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B
        using a custom CUDA kernel.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return self.tril_matmul_op(A, B)