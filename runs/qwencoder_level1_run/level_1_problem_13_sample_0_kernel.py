import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for symmetric matrix multiplication
symmetric_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void symmetric_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros({N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 16;
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);
    
    symmetric_matmul_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
    
    return C;
}
"""

symmetric_matmul_cpp_source = (
    "torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for symmetric matrix multiplication
symmetric_matmul = load_inline(
    name="symmetric_matmul",
    cpp_sources=symmetric_matmul_cpp_source,
    cuda_sources=symmetric_matmul_source,
    functions=["symmetric_matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for symmetric matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.symmetric_matmul = symmetric_matmul
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices using custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return self.symmetric_matmul.symmetric_matmul_cuda(A, B)

N = 4096

def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    A = torch.rand(N, N, device='cuda')
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.rand(N, N, device='cuda')
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []