import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (A.T @ B)
# This kernel uses a tiled approach with shared memory for performance.
matmul_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define the tile dimension for shared memory blocking
#define TILE_DIM 32

// CUDA kernel to compute C = A.T @ B
// A is a (K, M) tensor
// B is a (K, N) tensor
// C is the output (M, N) tensor
__global__ void matmul_transpose_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory tiles for A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Thread indices
    int bx = blockIdx.x;  // Block index in x dimension
    int by = blockIdx.y;  // Block index in y dimension
    int tx = threadIdx.x; // Thread index in x dimension
    int ty = threadIdx.y; // Thread index in y dimension

    // Calculate the global row and column for the C matrix
    // This thread is responsible for computing C[row, col]
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    // Accumulator for the result
    float C_value = 0.0f;

    // Loop over the tiles along the K dimension
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Cooperatively load a tile of A.T and B into shared memory
        
        // Load element for As (tile of A.T)
        // We want A.T[row, t*TILE_DIM + tx], which is A[t*TILE_DIM + tx, row]
        int a_k = t * TILE_DIM + tx;
        int a_m = row;
        if (a_k < K && a_m < M) {
            As[ty][tx] = A[a_k * M + a_m];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load element for Bs (tile of B)
        // We want B[t*TILE_DIM + ty, col]
        int b_k = t * TILE_DIM + ty;
        int b_n = col;
        if (b_k < K && b_n < N) {
            Bs[ty][tx] = B[b_k * N + b_n];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all threads in the block have loaded their data
        __syncthreads();

        // Multiply the tiles from shared memory and accumulate the result
        for (int k = 0; k < TILE_DIM; ++k) {
            // A.T[row, t*TILE_DIM + k] is in As[ty][k]
            // B[t*TILE_DIM + k, col] is in Bs[k][tx]
            C_value += As[ty][k] * Bs[k][tx];
        }

        // Synchronize again before loading the next tile
        __syncthreads();
    }

    // Write the final result to the global memory C matrix
    if (row < M && col < N) {
        C[row * N + col] = C_value;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same size in dimension 0 (the K dimension)");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Both tensors must be on a CUDA device");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Both tensors must be contiguous");

    // Get dimensions
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(1);

    // Create the output tensor C with shape (M, N)
    auto C = torch::zeros({M, N}, A.options());

    // Configure grid and block dimensions
    const dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    const dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch the kernel
    matmul_transpose_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
    }

    return C;
}
"""

matmul_transpose_cpp_source = """
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B);
"""

# JIT compile the CUDA kernel
matmul_transpose_lib = load_inline(
    name="matmul_transpose_lib",
    cpp_sources=matmul_transpose_cpp_source,
    cuda_sources=matmul_transpose_source,
    functions=["matmul_transpose_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A.T * B)
    using a custom tiled CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled CUDA function
        self.custom_matmul = matmul_transpose_lib.matmul_transpose_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using the custom CUDA kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # The custom kernel is designed to compute A.T @ B directly
        return self.custom_matmul(A, B)