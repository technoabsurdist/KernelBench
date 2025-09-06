import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tiled matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// TILE_DIM must be a multiple of warpSize (32) for best performance
#define TILE_DIM 32

// CUDA kernel for tiled matrix multiplication (C = A * B)
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column for the current thread
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    // Accumulator for the C element
    float C_val = 0.0f;

    // Loop over the tiles of A and B required to compute the C tile
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load a tile of A into shared memory
        if (row < M && (t * TILE_DIM + tx) < K) {
            As[ty][tx] = A[row * K + (t * TILE_DIM + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load a tile of B into shared memory
        if (col < N && (t * TILE_DIM + ty) < K) {
            Bs[ty][tx] = B[(t * TILE_DIM + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Multiply the two tiles and accumulate the result
        for (int k = 0; k < TILE_DIM; ++k) {
            C_val += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the final result to global memory
    if (row < M && col < N) {
        C[row * N + col] = C_val;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check input tensors
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions are not compatible for multiplication");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be on CUDA device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "Input tensors must be of type float32");

    // Get dimensions
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Define grid and block dimensions
    const dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    const dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch the kernel
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# JIT compile the custom CUDA kernel
custom_matmul_module = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Model that performs a single matrix multiplication (C = A * B)
    using a custom tiled CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled module
        self.custom_matmul = custom_matmul_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using the custom CUDA kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Ensure inputs are on the correct device and contiguous
        A_cuda = A.cuda().contiguous()
        B_cuda = B.cuda().contiguous()
        
        return self.custom_matmul.matmul_cuda(A_cuda, B_cuda)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed