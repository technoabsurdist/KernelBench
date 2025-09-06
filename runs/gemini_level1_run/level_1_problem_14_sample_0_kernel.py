import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for upper triangular matrix multiplication
triu_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_DIM 32

__global__ void triu_matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Global row and column for the C element this thread's block computes
    const int row = by * TILE_DIM + ty;
    const int col = bx * TILE_DIM + tx;

    // Early exit for threads that would compute the lower triangular part of C
    if (row > col) {
        return;
    }

    // Shared memory for tiles of A and B
    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM];

    // Accumulator for the C element
    float C_val = 0.0f;

    // Loop over tiles of A and B to compute the dot product
    for (int k_tile = 0; k_tile < (N + TILE_DIM - 1) / TILE_DIM; ++k_tile) {
        // --- Load A_tile ---
        // Global coordinates for loading A tile
        const int a_load_row = by * TILE_DIM + ty;
        const int a_load_col = k_tile * TILE_DIM + tx;

        // Load from A if within bounds and in the upper triangle of A, otherwise load 0
        if (a_load_row < N && a_load_col < N && a_load_row <= a_load_col) {
            A_tile[ty][tx] = A[a_load_row * N + a_load_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        // --- Load B_tile ---
        // Global coordinates for loading B tile
        const int b_load_row = k_tile * TILE_DIM + ty;
        const int b_load_col = bx * TILE_DIM + tx;

        // Load from B if within bounds and in the upper triangle of B, otherwise load 0
        if (b_load_row < N && b_load_col < N && b_load_row <= b_load_col) {
            B_tile[ty][tx] = B[b_load_row * N + b_load_col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        // Synchronize to make sure all data is loaded into shared memory
        __syncthreads();

        // --- Compute dot product for the tile ---
        for (int k = 0; k < TILE_DIM; ++k) {
            C_val += A_tile[ty][k] * B_tile[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the final result to global memory if within bounds
    if (row < N && col < N) {
        C[row * N + col] = C_val;
    }
}

torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.sizes() == B.sizes(), "A and B must have the same shape");
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    const int N = A.size(0);

    // Create output tensor, initialized to zeros
    auto C = torch::zeros_like(A);

    // Define grid and block dimensions
    const dim3 block_dim(TILE_DIM, TILE_DIM);
    const dim3 grid_dim(
        (N + TILE_DIM - 1) / TILE_DIM,
        (N + TILE_DIM - 1) / TILE_DIM
    );

    // Launch the kernel
    triu_matmul_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

triu_matmul_cpp_source = (
    "torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for triangular matrix multiplication
triu_matmul = load_inline(
    name="triu_matmul",
    cpp_sources=triu_matmul_cpp_source,
    cuda_sources=triu_matmul_source,
    functions=["triu_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B) for upper triangular matrices
    using a custom CUDA kernel. The kernel is fused to compute only the upper triangular
    part of the result, avoiding redundant computations.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.triu_matmul = triu_matmul

    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using a custom CUDA kernel.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return self.triu_matmul.triu_matmul_cuda(A, B)