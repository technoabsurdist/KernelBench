import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication using shared memory for tiling
bmm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void bmm_kernel_shared_mem(const float* A, const float* B, float* C, int m, int k, int n) {
    // Get batch index from the third dimension of the grid
    int batch_idx = blockIdx.z;

    // Pointers to the start of the matrices for this specific batch item
    const float* A_batch = A + batch_idx * m * k;
    const float* B_batch = B + batch_idx * k * n;
    float* C_batch = C + batch_idx * m * n;

    // Identify the row and column of the C tile to work on
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Identify the thread's position within the block
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Allocate shared memory for tiles of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Each thread computes one element of the C sub-matrix. Initialize it.
    float C_value = 0.0f;

    // Loop over the tiles of A and B required to compute the C sub-matrix
    for (int tile_idx = 0; tile_idx < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++tile_idx) {
        // Load a tile of A into shared memory
        int a_row = block_row * TILE_WIDTH + thread_row;
        int a_col = tile_idx * TILE_WIDTH + thread_col;
        if (a_row < m && a_col < k) {
            As[thread_row][thread_col] = A_batch[a_row * k + a_col];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        // Load a tile of B into shared memory
        int b_row = tile_idx * TILE_WIDTH + thread_row;
        int b_col = block_col * TILE_WIDTH + thread_col;
        if (b_row < k && b_col < n) {
            Bs[thread_row][thread_col] = B_batch[b_row * n + b_col];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded before computation
        __syncthreads();

        // Multiply the two tiles and accumulate the result
        for (int i = 0; i < TILE_WIDTH; ++i) {
            C_value += As[thread_row][i] * Bs[i][thread_col];
        }

        // Synchronize before loading the next tiles
        __syncthreads();
    }

    // Write the final result to global memory
    int c_row = block_row * TILE_WIDTH + thread_row;
    int c_col = block_col * TILE_WIDTH + thread_col;
    if (c_row < m && c_col < n) {
        C_batch[c_row * n + c_col] = C_value;
    }
}

torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dim() == 3, "A must be a 3D tensor");
    TORCH_CHECK(B.dim() == 3, "B must be a 3D tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match (A.shape[2] vs B.shape[1])");

    // Get dimensions
    const auto batch_size = A.size(0);
    const auto m = A.size(1);
    const auto k = A.size(2);
    const auto n = B.size(2);

    // Create output tensor
    auto C = torch::zeros({batch_size, m, n}, A.options());

    // Define grid and block dimensions
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks(
        (n + TILE_WIDTH - 1) / TILE_WIDTH,
        (m + TILE_WIDTH - 1) / TILE_WIDTH,
        batch_size
    );

    // Launch the kernel
    bmm_kernel_shared_mem<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        m, k, n
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

bmm_cpp_source = "torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
bmm_cuda_module = load_inline(
    name="bmm_cuda_module",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_source,
    functions=["bmm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_cuda = bmm_cuda_module.bmm_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication with a custom kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return self.bmm_cuda(A, B)