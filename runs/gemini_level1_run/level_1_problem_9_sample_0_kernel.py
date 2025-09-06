import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tiled matrix multiplication
# This is a standard optimization technique that uses shared memory to reduce global memory access.
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define the tile width for shared memory. 32 is a good default as it matches the warp size.
#define TILE_WIDTH 32

// CUDA kernel for C = A * B
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory tiles for A and B
    __shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column for the C element this thread will compute
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Accumulator for the result
    float c_val = 0.0f;

    // Loop over the tiles in the K dimension
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load a tile of A into shared memory
        // Each thread in the block loads one element of the tile
        int a_row_idx = row;
        int a_col_idx = t * TILE_WIDTH + tx;
        if (a_row_idx < M && a_col_idx < K) {
            a_tile[ty][tx] = A[a_row_idx * K + a_col_idx];
        } else {
            a_tile[ty][tx] = 0.0f;
        }

        // Load a tile of B into shared memory
        // Each thread in the block loads one element of the tile
        int b_row_idx = t * TILE_WIDTH + ty;
        int b_col_idx = col;
        if (b_row_idx < K && b_col_idx < N) {
            b_tile[ty][tx] = B[b_row_idx * N + b_col_idx];
        } else {
            b_tile[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all data is loaded into shared memory before computation
        __syncthreads();

        // Perform the matrix multiplication for the current tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            c_val += a_tile[ty][k] * b_tile[k][tx];
        }

        // Synchronize before loading the next tiles
        __syncthreads();
    }

    // Write the final result to the output matrix C in global memory
    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}

// C++ wrapper to be called from Python
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Validate inputs
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions are not compatible for multiplication");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on a CUDA device");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // Create the output tensor
    auto C = torch::empty({M, N}, A.options());

    // Configure the grid and block dimensions for the kernel launch
    const dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    const dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the kernel
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

# C++ source for the function declaration (needed by load_inline)
matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# JIT compile the custom CUDA kernel
custom_matmul = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for the matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled CUDA function
        self.custom_matmul_op = custom_matmul.matmul_cuda

    def forward(self, A, B):
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K).
            B (torch.Tensor): Input matrix of shape (K, N).

        Returns:
            torch.Tensor: Output matrix of shape (M, N).
        """
        # The custom kernel requires contiguous tensors.
        # While the inputs from get_inputs are contiguous, this is good practice.
        A_cont = A.contiguous()
        B_cont = B.contiguous()
        return self.custom_matmul_op(A_cont, B_cont)