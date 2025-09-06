import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto A_sizes = A.sizes();
    auto B_sizes = B.sizes();
    
    int M = A_sizes[0];
    int K = A_sizes[1];
    int N = B_sizes[1];
    
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // For tall-skinny matrices, use a specialized approach
    if (M > 1000 * N || N > 1000 * K) {
        const int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        
        matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    } else {
        // Use cuBLAS for general case
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, M, K,
                    &alpha, 
                    B.data_ptr<float>(), N,
                    A.data_ptr<float>(), K,
                    &beta, 
                    C.data_ptr<float>(), N);
        
        cublasDestroy(handle);
    }
    
    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication with custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return self.matmul.matmul_cuda(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed