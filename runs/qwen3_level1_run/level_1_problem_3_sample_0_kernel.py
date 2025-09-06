import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
bmm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, 
                                      int batch_size, int m, int n, int k) {
    // Get batch index
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    // Calculate indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[batch_idx * m * k + row * k + i] * B[batch_idx * k * n + i * n + col];
        }
        C[batch_idx * m * n + row * n + col] = sum;
    }
}

torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    auto C = torch::zeros({batch_size, m, n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Define block and grid dimensions
    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x, 
                   (m + block_size.y - 1) / block_size.y,
                   batch_size);
    
    batched_matmul_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        batch_size, m, n, k
    );
    
    return C;
}
"""

bmm_cpp_source = (
    "torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for batched matrix multiplication
bmm_module = load_inline(
    name="bmm_cuda",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_cuda_source,
    functions=["bmm_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized batched matrix multiplication using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_func = bmm_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using custom CUDA kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        # Ensure tensors are on CUDA and have correct dtype
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        if A.dtype != torch.float32:
            A = A.float()
        if B.dtype != torch.float32:
            B = B.float()
            
        return self.bmm_func.bmm_cuda(A, B)

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed