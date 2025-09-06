import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (C = A * B.T)
# This kernel uses tiling with shared memory to improve performance.
matmul_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// The width of the tiles processed by each thread block.
// Should be a multiple of the warp size (32) for best performance.
#define TILE_WIDTH 32

// CUDA kernel to compute C = A * B.T
// A is a (M, K) matrix
// B is a (N, K) matrix
// C is the output (M, N) matrix
__global__ void matmul_transpose_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Thread indices
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Global row and column for this thread's output element in C
    int global_row = block_row * TILE_WIDTH + thread_row;
    int global_col = block_col * TILE_WIDTH + thread_col;

    // Accumulator for the dot product, stored in a register
    float C_value = 0.0f;

    // Loop over the tiles of A and B along the K dimension
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Cooperatively load a tile of A into shared memory
        int a_col = t * TILE_WIDTH + thread_col;
        if (global_row < M && a_col < K) {
            As[thread_row][thread_col] = A[global_row * K + a_col];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        // Cooperatively load a tile of B into shared memory.
        // This corresponds to the tile of B.T that we need.
        int b_col = t * TILE_WIDTH + thread_col;
        if (global_col < N && b_col < K) {
            Bs[thread_row][thread_col] = B[global_col * K + b_col];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

        // Synchronize to ensure all threads in the block have loaded their data
        __syncthreads();

        // Multiply the two tiles from shared memory and accumulate the result
        for (int k = 0; k < TILE_WIDTH; ++k) {
            C_value += As[thread_row][k] * Bs[thread_col][k];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the final result to the global memory output matrix C
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = C_value;
    }
}

// C++ wrapper function to be called from PyTorch
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions of A and B.T must match (A.shape[1] == B.shape[1])");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on a CUDA device");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");


    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    // Create the output tensor C with shape (M, N)
    auto C = torch::zeros({M, N}, A.options());

    // Define grid and block dimensions for the kernel launch
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the CUDA kernel
    matmul_transpose_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

# C++ source for function signature declaration
matmul_transpose_cpp_source = "torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
# This creates a Python module with the matmul_transpose_cuda function.
matmul_transpose = load_inline(
    name="matmul_transpose",
    cpp_sources=matmul_transpose_cpp_source,
    cuda_sources=matmul_transpose_source,
    functions=["matmul_transpose_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for the matrix multiplication.
    The kernel computes C = A * B.T directly.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # The compiled CUDA function is attached to the model instance
        self.custom_matmul = matmul_transpose.matmul_transpose_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using the custom CUDA kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N), result of A @ B.T.
        """
        # Ensure inputs are contiguous for the custom kernel
        A_cont = A.contiguous()
        B_cont = B.contiguous()
        return self.custom_matmul(A_cont, B_cont)