import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;
    if (idx < total_elements) {
        int row = idx / cols;
        int col = idx % cols;
        output[col * rows + row] = input[idx];
    }
}

torch::Tensor matmul_custom_cuda(torch::Tensor A, torch::Tensor B) {
    // A is (M, K), B is (N, K) -> we want A @ B.T which is (M, K) @ (K, N) -> (M, N)
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(0);
    
    // First transpose B from (N, K) to (K, N)
    auto B_T = torch::zeros({K, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int num_blocks_B = (N * K + block_size - 1) / block_size;
    transpose_kernel<<<num_blocks_B, block_size>>>(B.data_ptr<float>(), B_T.data_ptr<float>(), N, K);
    
    // Perform matrix multiplication: (M, K) @ (K, N) -> (M, N)
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cublasSgemm: C = alpha * A * B + beta * C
    // A: (M, K), B: (K, N), C: (M, N)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K,
                &alpha,
                B_T.data_ptr<float>(), N,
                A.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);
    
    cublasDestroy(handle);
    
    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_custom_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matmul_custom = load_inline(
    name="matmul_custom",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_custom_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_custom = matmul_custom
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using custom CUDA kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_custom.matmul_custom_cuda(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K, device='cuda')
    B = torch.rand(N, K, device='cuda')
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed