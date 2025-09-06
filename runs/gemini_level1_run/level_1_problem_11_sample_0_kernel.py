import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Tiled matrix multiplication kernel to compute C = A * B
// A is treated as a 2D matrix of shape (M, N)
// B is a 2D matrix of shape (N, P)
// C is a 2D matrix of shape (M, P)
// where M = b*i*j, N = l, P = k from the original problem
__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int P) {
    // Shared memory for tiles of A and B
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    // Thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column for this thread's output element
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float C_value = 0.0f;

    // Loop over tiles along the N dimension
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load A_tile from global memory to shared memory
        // Each thread loads one element
        int A_col = t * TILE_WIDTH + tx;
        if (row < M && A_col < N) {
            A_tile[ty][tx] = A[row * N + A_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        // Load B_tile from global memory to shared memory
        // Each thread loads one element
        int B_row = t * TILE_WIDTH + ty;
        if (col < P && B_row < N) {
            B_tile[ty][tx] = B[B_row * P + col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute dot product for the tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            C_value += A_tile[ty][k] * B_tile[k][tx];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < P) {
        C[row * P + col] = C_value;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // A: (b, i, j, l)
    // B: (l, k)
    // C: (b, i, j, k)

    TORCH_CHECK(A.dim() == 4, "A must be a 4D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(3) == B.size(0), "Inner dimensions must match: A.shape[3] vs B.shape[0]");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");

    const auto b = A.size(0);
    const auto i = A.size(1);
    const auto j = A.size(2);
    const auto l = A.size(3);
    const auto k = B.size(1);

    // Flatten the first three dimensions of A for the matmul
    const int M = b * i * j;
    const int N = l;
    const int P = k;

    auto C = torch::empty({b, i, j, k}, A.options());

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((P + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    batched_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, P
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return C;
}
"""

batched_matmul_cpp_source = (
    "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Performs 4D tensor-matrix multiplication using a custom CUDA kernel:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_matmul = batched_matmul.batched_matmul_cuda

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication using the custom kernel.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return self.custom_matmul(A, B)