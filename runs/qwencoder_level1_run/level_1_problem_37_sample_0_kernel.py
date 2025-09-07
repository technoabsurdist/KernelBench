import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Frobenius norm normalization
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void square_elements_kernel(const float* input, float* squared, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        squared[idx] = input[idx] * input[idx];
    }
}

__global__ void normalize_kernel(const float* input, float* output, float norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    
    // Allocate temporary memory for squared values
    float* d_squared;
    cudaMalloc(&d_squared, size * sizeof(float));
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // Square all elements
    square_elements_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), d_squared, size);
    
    // Reduce to compute sum of squares
    // Use CUB for reduction
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary device storage requirements
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_squared, d_squared, size);
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_squared, d_squared, size);
    
    // Copy the result back to host to get the sum
    float sum_squares;
    cudaMemcpy(&sum_squares, d_squared, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute Frobenius norm (sqrt of sum of squares)
    float norm = sqrtf(sum_squares);
    
    // Normalize the input tensor
    normalize_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), norm, size);
    
    // Cleanup
    cudaFree(d_squared);
    cudaFree(d_temp_storage);
    
    return output;
}
"""

frobenius_norm_cpp_source = (
    "torch::Tensor frobenius_norm_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Frobenius norm normalization
frobenius_norm = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return self.frobenius_norm.frobenius_norm_cuda(x)