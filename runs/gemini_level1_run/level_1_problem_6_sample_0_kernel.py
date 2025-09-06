import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tiled matrix multiplication
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// TILE_WIDTH must be a power of 2, 16 or 32 are common choices
#define TILE_WIDTH 16

// CUDA kernel for tiled matrix multiplication: C = A * B
// A is (M, K), B is (K, N), C is (M, N)
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];

    // Identify the row and column of the C tile to compute
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    // Identify the thread's position within the C tile
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Calculate the global row and column for the C element
    int global_row = tile_row * TILE_WIDTH + thread_row;
    int global_col = tile_col * TILE_WIDTH + thread_col;

    // Accumulator for the C element
    float acc = 0.0f;

    // Loop over the tiles of A and B required to compute the C tile
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load a tile of A into shared memory
        int a_col = t * TILE_WIDTH + thread_col;
        if (global_row < M && a_col < K) {
            a_tile[thread_row][thread_col] = A[global_row * K + a_col];
        } else {
            a_tile[thread_row][thread_col] = 0.0f;
        }

        // Load a tile of B into shared memory
        int b_row = t * TILE_WIDTH + thread_row;
        if (b_row < K && global_col < N) {
            b_tile[thread_row][thread_col] = B[b_row * N + global_col];
        } else {
            b_tile[thread_row][thread_col] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Multiply the tiles and accumulate the results
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += a_tile[thread_row][k] * b_tile[k][thread_col];
        }

        // Synchronize before loading the next tiles
        __syncthreads();
    }

    // Write the final result to global memory
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = acc;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure tensors are on the GPU and are contiguous
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");

    // Get dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    TORCH_CHECK(K == B.size(0), "Matrix dimensions are not compatible for multiplication");

    // Create the output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Define grid and block dimensions
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Launch the kernel
    matmul_kernel<<<grid_dim, block_dim>>>(
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

# JIT compile the CUDA kernel
custom_matmul_lib = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled library function
        self.custom_matmul = custom_matmul_lib.matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using the custom CUDA kernel.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        # The custom kernel expects contiguous tensors
        A_cont = A.contiguous()
        B_cont = B.contiguous()
        return self.custom_matmul(A_cont, B_cont)

M = 256
N = 256
K = 131072 * 4

def get_inputs():
    # Use float32 as the custom kernel is written for it
    A = torch.rand(M, K, dtype=torch.float32).cuda()
    B = torch.rand(K, N, dtype=torch.float32).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed