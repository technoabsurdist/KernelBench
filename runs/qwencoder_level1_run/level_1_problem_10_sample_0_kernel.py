import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D tensor-matrix multiplication
tensor_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void tensor_matmul_kernel(const float* A, const float* B, float* C, 
                                     int N, int M, int K, int L) {
    // Calculate which matrix in the batch we're processing
    int n = blockIdx.z;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n < N && m < M && l < L) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[(n * M * K) + (m * K) + k] * B[k * L + l];
        }
        C[(n * M * L) + (m * L) + l] = sum;
    }
}

torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);
    
    auto C = torch::zeros({N, M, L}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch configuration
    const int BLOCK_SIZE_M = 16;
    const int BLOCK_SIZE_L = 16;
    
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((L + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y, 
                  N);
    
    tensor_matmul_kernel<<<gridSize, blockSize>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        N, M, K, L
    );
    
    return C;
}
"""

tensor_matmul_cpp_source = (
    "torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for tensor-matrix multiplication
tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_source,
    functions=["tensor_matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication with custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matmul = tensor_matmul
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication with custom CUDA kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return self.tensor_matmul.tensor_matmul_cuda(A, B)

N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    A = torch.rand(N, M, K).cuda()
    B = torch.rand(K, L).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed