import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tiled matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

// Tiled matrix multiplication kernel for square matrices (N x N)
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column for the output element C this thread will compute
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Accumulator for the output element
    float C_val = 0.0f;

    // Loop over the tiles of A and B required to compute the C tile
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Load tile of A into shared memory
        // Global index for A: A[row, m * TILE_WIDTH + tx]
        if (row < N && (m * TILE_WIDTH + tx) < N) {
            As[ty][tx] = A[row * N + (m * TILE_WIDTH + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        // Global index for B: B[m * TILE_WIDTH + ty, col]
        if ((m * TILE_WIDTH + ty) < N && col < N) {
            Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to make sure all threads in the block have loaded their data
        __syncthreads();

        // Multiply the two tiles and accumulate the result in the register
        for (int k = 0; k < TILE_WIDTH; ++k) {
            C_val += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure all threads are done with the current tiles
        // before loading the next ones
        __syncthreads();
    }

    // Write the final result from the register to global memory
    if (row < N && col < N) {
        C[row * N + col] = C_val;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must be compatible for multiplication");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");

    const int N = A.size(0);

    // Create the output tensor
    auto C = torch::zeros_like(A);

    // Define grid and block dimensions
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the kernel
    matmul_tiled_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication
# This is done once when the Python module is imported.
matmul_lib = load_inline(
    name="matmul_lib",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom tiled matrix multiplication CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        # The C++ wrapper function already contains checks for shape, device, and dtype.
        return matmul_lib.matmul_cuda(A, B)