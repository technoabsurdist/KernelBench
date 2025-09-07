import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tall-skinny matrix multiplication
tall_skinny_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void tall_skinny_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Each thread computes one element of the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor tall_skinny_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto A_sizes = A.sizes();
    auto B_sizes = B.sizes();
    
    int M = A_sizes[0];
    int K = A_sizes[1];
    int N = B_sizes[1];
    
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch configuration
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    tall_skinny_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    
    return C;
}
"""

tall_skinny_matmul_cpp_source = (
    "torch::Tensor tall_skinny_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for tall-skinny matrix multiplication
tall_skinny_matmul = load_inline(
    name="tall_skinny_matmul",
    cpp_sources=tall_skinny_matmul_cpp_source,
    cuda_sources=tall_skinny_matmul_source,
    functions=["tall_skinny_matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for tall-skinny matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tall_skinny_matmul = tall_skinny_matmul
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) where M >> K.
            B (torch.Tensor): Input matrix of shape (K, N) where K << N.

        Returns:
            torch.Tensor: Output matrix of shape (M, N)
        """
        return self.tall_skinny_matmul.tall_skinny_matmul_cuda(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed