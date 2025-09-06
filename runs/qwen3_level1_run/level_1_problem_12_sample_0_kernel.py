import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for diagonal matrix multiplication
diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* diag, const float* B, float* C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        C[row * M + col] = diag[row] * B[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = B.size(1);
    
    auto C = torch::zeros_like(B);
    
    const int block_size_x = 16;
    const int block_size_y = 16;
    
    dim3 block(block_size_x, block_size_y);
    dim3 grid((M + block_size_x - 1) / block_size_x, (N + block_size_y - 1) / block_size_y);
    
    diag_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);
    
    return C;
}
"""

diag_matmul_cpp_source = (
    "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for diagonal matrix multiplication
diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for diagonal matrix multiplication.
    C = diag(A) * B
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using custom CUDA kernel.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return self.diag_matmul.diag_matmul_cuda(A, B)

M = 4096
N = 4096

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed