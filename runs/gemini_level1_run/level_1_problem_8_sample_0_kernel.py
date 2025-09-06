import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tiled matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// TILE_WIDTH must be a multiple of warpSize (32) for best performance.
// 32 is a good default, but 16 can also be effective.
#define TILE_WIDTH 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column for the C element this thread block is computing
    int row_start = by * TILE_WIDTH;
    int col_start = bx * TILE_WIDTH;

    // The specific C element this thread will compute
    int C_row = row_start + ty;
    int C_col = col_start + tx;

    // Accumulator for the C element, stored in a register
    float C_value = 0.0f;

    // Loop over the tiles of A and B required to compute the C tile
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // --- Load tile of A into shared memory ---
        int A_col = t * TILE_WIDTH + tx;
        if (C_row < M && A_col < K) {
            As[ty][tx] = A[C_row * K + A_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // --- Load tile of B into shared memory ---
        int B_row = t * TILE_WIDTH + ty;
        if (C_col < N && B_row < K) {
            Bs[ty][tx] = B[B_row * N + C_col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to make sure all threads in the block have loaded their data
        __syncthreads();

        // --- Multiply the tiles from shared memory and accumulate ---
        for (int k = 0; k < TILE_WIDTH; ++k) {
            C_value += As[ty][k] * Bs[k][tx];
        }

        // Synchronize again before loading the next tiles
        __syncthreads();
    }

    // Write the final result to global memory C, with boundary check
    if (C_row < M && C_col < N) {
        C[C_row * N + C_col] = C_value;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Input tensors must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions are not compatible for multiplication (A.shape[1] != B.shape[0])");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be on a CUDA device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "Input tensors must be of type float32");

    // Ensure tensors are contiguous in memory for correct pointer arithmetic
    A = A.contiguous();
    B = B.contiguous();

    // Get matrix dimensions
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // Create the output tensor C on the same device as A
    auto C = torch::zeros({M, N}, A.options());

    // Define grid and block dimensions for the kernel launch
    const dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid_dim(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Launch the CUDA kernel
    matmul_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        K,
        N
    );

    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication
# This happens only once when the Python module is loaded.
matmul_cuda_kernel = load_inline(
    name="matmul_cuda_kernel",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled kernel function
        self.matmul_op = matmul_cuda_kernel.matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using a custom CUDA kernel.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        # Ensure inputs are on the GPU before passing to the CUDA kernel
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
            
        return self.matmul_op(A, B)