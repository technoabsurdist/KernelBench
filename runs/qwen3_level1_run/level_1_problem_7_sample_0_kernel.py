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
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // For small K, we can use a simple kernel
    if (K <= 64) {
        dim3 block_size(16, 16);
        dim3 num_blocks((N + block_size.x - 1) / block_size.x, 
                        (M + block_size.y - 1) / block_size.y);
        
        matmul_kernel<<<num_blocks, block_size>>>(
            A.data_ptr<float>(), 
            B.data_ptr<float>(), 
            C.data_ptr<float>(), 
            M, N, K
        );
    } else {
        // For larger K, use cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
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
    "torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);"
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
    Optimized model with custom CUDA matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_func = matmul
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Ensure tensors are on GPU and have correct dtype
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        if A.dtype != torch.float32:
            A = A.float()
        if B.dtype != torch.float32:
            B = B.float()
            
        return self.matmul_func.matmul_cuda(A, B)

M = 16384 * 2
N = 16384 * 2
K = 32 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed