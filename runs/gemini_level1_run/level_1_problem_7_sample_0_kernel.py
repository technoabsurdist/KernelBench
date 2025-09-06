import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tiled matrix multiplication
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// TILE_DIM determines the size of the tile processed by each thread block.
// A tile size of 16x16 means each block has 16*16 = 256 threads.
#define TILE_DIM 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Each thread block computes one tile of the output matrix C.
    // Each thread within the block computes one element of the tile.

    // Shared memory to store tiles of A and B, reducing global memory access.
    __shared__ float a_tile[TILE_DIM][TILE_DIM];
    __shared__ float b_tile[TILE_DIM][TILE_DIM];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column for the output element this thread will compute
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    // Accumulator for the output element, stored in a register.
    float c_val = 0.0f;

    // Loop over the tiles along the K dimension.
    for (int k_tile_idx = 0; k_tile_idx < (K + TILE_DIM - 1) / TILE_DIM; ++k_tile_idx) {
        // Each thread loads one element of the A tile into shared memory.
        int a_row = blockIdx.y * TILE_DIM + ty;
        int a_col = k_tile_idx * TILE_DIM + tx;
        if (a_row < M && a_col < K) {
            a_tile[ty][tx] = A[a_row * K + a_col];
        } else {
            a_tile[ty][tx] = 0.0f; // Pad with zero if out of bounds
        }

        // Each thread loads one element of the B tile into shared memory.
        int b_row = k_tile_idx * TILE_DIM + ty;
        int b_col = blockIdx.x * TILE_DIM + tx;
        if (b_row < K && b_col < N) {
            b_tile[ty][tx] = B[b_row * N + b_col];
        } else {
            b_tile[ty][tx] = 0.0f; // Pad with zero if out of bounds
        }

        // Synchronize to ensure all data is loaded into shared memory before computation.
        __syncthreads();

        // Perform the matrix multiplication for the current tiles.
        // Each thread computes its part of the dot product.
        for (int k = 0; k < TILE_DIM; ++k) {
            c_val += a_tile[ty][k] * b_tile[k][tx];
        }

        // Synchronize to ensure all computations with the current tiles are finished
        // before loading the next tiles.
        __syncthreads();
    }

    // Write the final accumulated value to the output matrix C in global memory.
    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}

// C++ wrapper function that will be called from Python.
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation checks.
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions of A and B must match for matrix multiplication");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Both tensors must be on a CUDA device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");

    // Get matrix dimensions.
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // Create the output tensor C with the correct shape and device.
    auto C = torch::zeros({M, N}, A.options());

    // Define the kernel launch configuration.
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch the CUDA kernel.
    matmul_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

# C++ source for the function signature, required by load_inline.
matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code.
matmul_module = load_inline(
    name="matmul_module",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the custom matmul function
        self.custom_matmul = matmul_module.matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using the custom CUDA kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return self.custom_matmul(A, B)