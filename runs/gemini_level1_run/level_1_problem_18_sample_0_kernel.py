import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the transposed matrix multiplication
# This kernel computes C = A.T @ B.T where A is (K, M) and B is (N, K)
# The resulting C is (M, N).
# The computation is C[m, n] = sum_k A[k, m] * B[n, k]
matmul_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// TILE_WIDTH determines the size of the tile processed by each thread block.
// A value of 32 is a common choice that balances parallelism and resource usage.
#define TILE_WIDTH 32

__global__ void matmul_transpose_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Identify the row (m) and column (n) of the C matrix this thread will compute.
    const int m = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int n = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Shared memory to hold tiles of A and B. This reduces global memory access.
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    float C_val = 0.0f;

    // Loop over the K dimension in tiles.
    for (int k_tile = 0; k_tile < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++k_tile) {
        const int k_tile_base = k_tile * TILE_WIDTH;

        // --- Load a tile of A into shared memory ---
        // Each thread in the block loads one element of the A tile.
        // A is (K, M), we need A[k, m].
        // Thread (threadIdx.y, threadIdx.x) loads A[k_base + threadIdx.y, m_base + threadIdx.x]
        const int m_tile_base = blockIdx.y * TILE_WIDTH;
        const int a_k = k_tile_base + threadIdx.y;
        const int a_m = m_tile_base + threadIdx.x;
        if (a_k < K && a_m < M) {
            As[threadIdx.y][threadIdx.x] = A[a_k * M + a_m];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f; // Pad with zero for out-of-bounds accesses
        }

        // --- Load a tile of B into shared memory ---
        // Each thread in the block loads one element of the B tile.
        // B is (N, K), we need B[n, k].
        // Thread (threadIdx.y, threadIdx.x) loads B[n_base + threadIdx.y, k_base + threadIdx.x]
        const int n_tile_base = blockIdx.x * TILE_WIDTH;
        const int b_n = n_tile_base + threadIdx.y;
        const int b_k = k_tile_base + threadIdx.x;
        if (b_n < N && b_k < K) {
            Bs[threadIdx.y][threadIdx.x] = B[b_n * K + b_k];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f; // Pad with zero
        }

        // Synchronize to ensure all data is loaded into shared memory before proceeding.
        __syncthreads();

        // --- Compute the dot product for the tile ---
        // Each thread accumulates the product of a row from As and a column from Bs.
        // C[m, n] += sum_{k_inner} A[k_base+k_inner, m] * B[n, k_base+k_inner]
        // A[k_base+k_inner, m] is in As[k_inner][threadIdx.y]
        // B[n, k_base+k_inner] is in Bs[threadIdx.x][k_inner]
        for (int k_inner = 0; k_inner < TILE_WIDTH; ++k_inner) {
            C_val += As[k_inner][threadIdx.y] * Bs[threadIdx.x][k_inner];
        }

        // Synchronize to ensure all calculations for this tile are done before loading the next.
        __syncthreads();
    }

    // Write the final accumulated value to the output matrix C in global memory.
    if (m < M && n < N) {
        C[m * N + n] = C_val;
    }
}

// C++ wrapper function that will be callable from Python.
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation checks.
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");

    // Get dimensions from input tensors.
    // A has shape (K, M)
    // B has shape (N, K)
    const int K_A = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    const int K_B = B.size(1);

    TORCH_CHECK(K_A == K_B, "Inner dimensions of A.T and B.T must match (A.shape[0] == B.shape[1])");
    const int K = K_A;

    // Create the output tensor C with shape (M, N).
    auto C = torch::zeros({M, N}, A.options());

    // Define grid and block dimensions for the CUDA kernel launch.
    const dim3 threads(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the kernel.
    matmul_transpose_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    // Check for any CUDA errors during kernel execution.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

matmul_transpose_cpp_source = (
    "torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code. This will be done once when the Python module is loaded.
matmul_transpose = load_inline(
    name="matmul_transpose",
    cpp_sources=matmul_transpose_cpp_source,
    cuda_sources=matmul_transpose_source,
    functions=["matmul_transpose_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A.T * B.T)
    using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled CUDA function
        self.matmul_transpose_op = matmul_transpose.matmul_transpose_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using the custom CUDA kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_transpose_op(A, B)