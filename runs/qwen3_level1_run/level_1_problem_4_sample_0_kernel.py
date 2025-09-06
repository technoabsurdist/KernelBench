import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication
matvec_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void matvec_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k];
        }
        C[row] = sum;
    }
}

torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    
    auto C = torch::zeros({M, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int num_blocks = (M + block_size - 1) / block_size;
    
    matvec_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        M, 
        K
    );
    
    return C;
}
"""

matvec_cpp_source = (
    "torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix-vector multiplication
matvec = load_inline(
    name="matvec",
    cpp_sources=matvec_cpp_source,
    cuda_sources=matvec_source,
    functions=["matvec_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication with custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec = matvec
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using custom CUDA kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        # Ensure tensors are on CUDA and have correct dtype
        if A.device.type != 'cuda':
            A = A.cuda()
        if B.device.type != 'cuda':
            B = B.cuda()
        if A.dtype != torch.float32:
            A = A.float()
        if B.dtype != torch.float32:
            B = B.float()
            
        return self.matvec.matvec_cuda(A, B)

M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed