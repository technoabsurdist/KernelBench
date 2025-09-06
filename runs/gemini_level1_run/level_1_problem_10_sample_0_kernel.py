import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for batched matrix multiplication
bmm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// TILE_WIDTH should be a power of 2, e.g., 16 or 32.
// 32x32 = 1024 threads per block, which is often the max.
#define TILE_WIDTH 32

// CUDA kernel for batched matrix multiplication C = A * B
// A: (N, M, K), B: (K, L), C: (N, M, L)
// Grid: ( (L+TILE_WIDTH-1)/TILE_WIDTH, (M+TILE_WIDTH-1)/TILE_WIDTH, N )
// Block: ( TILE_WIDTH, TILE_WIDTH, 1 )
__global__ void bmm_kernel(const float* A, const float* B, float* C, int M, int K, int L) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Thread and block indices
    int batch = blockIdx.z;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Global row and column for the output C this thread is responsible for
    int C_row = block_row * TILE_WIDTH + thread_row;
    int C_col = block_col * TILE_WIDTH + thread_col;

    // Pointers to the start of the matrices for the current batch
    const float* A_batch = A + batch * M * K;
    float* C_batch = C + batch * M * L;

    // Accumulator for the C element
    float C_value = 0.0f;

    // Loop over the tiles of A and B in the K dimension
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load tile of A into shared memory
        int A_load_row = C_row;
        int A_load_col = t * TILE_WIDTH + thread_col;
        if (A_load_row < M && A_load_col < K) {
            As[thread_row][thread_col] = A_batch[A_load_row * K + A_load_col];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        // Load tile of B into shared memory
        int B_load_row = t * TILE_WIDTH + thread_row;
        int B_load_col = C_col;
        if (B_load_row < K && B_load_col < L) {
            Bs[thread_row][thread_col] = B[B_load_row * L + B_load_col];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Multiply the tiles from shared memory
        for (int k = 0; k < TILE_WIDTH; ++k) {
            C_value += As[thread_row][k] * Bs[k][thread_col];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    if (C_row < M && C_col < L) {
        C_batch[C_row * L + C_col] = C_value;
    }
}

torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.dim() == 3, "A must be a 3D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(2) == B.size(0), "Inner dimensions of A and B must match for matmul");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Both tensors must be on a CUDA device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "Both tensors must be of type float32");
    
    // Ensure tensors are contiguous for predictable memory layout
    A = A.contiguous();
    B = B.contiguous();

    // Get dimensions
    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    // Create output tensor
    auto C = torch::zeros({N, M, L}, A.options());

    // Define grid and block dimensions
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 numBlocks(
        (L + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH,
        N
    );

    // Launch the kernel
    bmm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, L
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    return C;
}
"""

bmm_cpp_source = "torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B);"

# JIT compile the custom CUDA kernel
bmm_matmul = load_inline(
    name="bmm_matmul",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_source,
    functions=["bmm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_matmul = bmm_matmul
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return self.bmm_matmul.bmm_cuda(A, B)