import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 4D tensor-matrix multiplication
tensor_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void tensor_matmul_kernel(const float* A, const float* B, float* C,
                                     int b, int i, int j, int l, int k) {
    // Calculate global thread indices
    int batch_idx = blockIdx.x;
    int i_idx = blockIdx.y;
    int j_idx = blockIdx.z;
    int k_idx = threadIdx.x;
    
    if (batch_idx >= b || i_idx >= i || j_idx >= j || k_idx >= k) return;
    
    // Compute dot product for C[batch_idx, i_idx, j_idx, k_idx]
    float sum = 0.0f;
    for (int l_idx = 0; l_idx < l; l_idx++) {
        sum += A[((batch_idx * i + i_idx) * j + j_idx) * l + l_idx] * 
               B[l_idx * k + k_idx];
    }
    
    C[((batch_idx * i + i_idx) * j + j_idx) * k + k_idx] = sum;
}

torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    auto A_sizes = A.sizes();
    int b = A_sizes[0];
    int i = A_sizes[1];
    int j = A_sizes[2];
    int l = A_sizes[3];
    
    auto B_sizes = B.sizes();
    int k = B_sizes[1];
    
    // Create output tensor
    auto C = torch::zeros({b, i, j, k}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Configure kernel launch parameters
    dim3 grid(b, i, j);
    dim3 block(min(k, 1024));
    
    // Launch kernel
    tensor_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(),
        b, i, j, l, k
    );
    
    return C;
}
"""

tensor_matmul_cpp_source = (
    "torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_source,
    functions=["tensor_matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Performs 4D tensor-matrix multiplication with custom CUDA kernel: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matmul = tensor_matmul

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication using custom CUDA kernel.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return self.tensor_matmul.tensor_matmul_cuda(A, B)

def get_inputs():
    A = torch.rand(b, i, j, l, device='cuda')
    B = torch.rand(l, k, device='cuda')
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed

# Test parameters
b = 8
i = 256
j = 512
l = 256
k = 768