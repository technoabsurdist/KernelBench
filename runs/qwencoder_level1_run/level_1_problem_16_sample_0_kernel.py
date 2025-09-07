import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

torch::Tensor matmul_custom_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(1);
    auto K = A.size(0);
    auto N = B.size(1);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto C = torch::zeros({M, N}, options);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Perform C = A^T * B using cuBLAS
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                N, M, K,
                &alpha,
                B.data_ptr<float>(), N,
                A.data_ptr<float>(), M,
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
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_custom.matmul_custom_cuda(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(K, M).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed