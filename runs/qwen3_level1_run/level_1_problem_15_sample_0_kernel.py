import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for lower triangular matrix multiplication
lower_tri_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void lower_tri_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col <= row) {  // Only compute for lower triangular elements
        float sum = 0.0f;
        for (int k = col; k <= row; k++) {  // Only iterate through valid range
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor lower_tri_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros_like(A);
    
    const int block_size = 16;
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);
    
    lower_tri_matmul_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
    
    return C;
}
"""

lower_tri_matmul_cpp_source = (
    "torch::Tensor lower_tri_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for lower triangular matrix multiplication
lower_tri_matmul = load_inline(
    name="lower_tri_matmul",
    cpp_sources=lower_tri_matmul_cpp_source,
    cuda_sources=lower_tri_matmul_source,
    functions=["lower_tri_matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of lower triangular matrices
    using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.lower_tri_matmul = lower_tri_matmul
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return self.lower_tri_matmul.lower_tri_matmul_cuda(A, B)

M = 4096

def get_inputs():
    A = torch.rand(M, M, device='cuda')
    B = torch.rand(M, M, device='cuda')
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed