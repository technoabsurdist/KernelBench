import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative sum (prefix sum)
cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void cumsum_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for block-level scan
    typedef cub::BlockScan<float, 1024> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    
    // Offset to the current batch
    const float* batch_input = input + batch_idx * seq_len;
    float* batch_output = output + batch_idx * seq_len;
    
    // Process elements in chunks
    for (int i = tid; i < seq_len; i += block_size) {
        float thread_data = (i < seq_len) ? batch_input[i] : 0.0f;
        
        // Perform inclusive prefix sum
        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
        __syncthreads();
        
        if (i < seq_len) {
            batch_output[i] = thread_data;
        }
    }
}

torch::Tensor cumsum_cuda(torch::Tensor input, int dim) {
    auto output = torch::zeros_like(input);
    auto sizes = input.sizes();
    
    if (dim < 0) dim += input.dim();
    
    if (dim == input.dim() - 1) {
        // Last dimension case - most common
        int batch_size = 1;
        for (int i = 0; i < input.dim() - 1; i++) {
            batch_size *= sizes[i];
        }
        int seq_len = sizes[input.dim() - 1];
        
        const int block_size = 1024;
        const int grid_size = batch_size;
        
        cumsum_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            batch_size, 
            seq_len
        );
    } else {
        // For other dimensions, fall back to PyTorch implementation
        // This is a simplified version - in practice you'd want to handle all dims
        return torch::cumsum(input, dim);
    }
    
    return output;
}
"""

cumsum_cpp_source = """
torch::Tensor cumsum_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for cumulative sum
cumsum = load_inline(
    name="cumsum",
    cpp_sources=cumsum_cpp_source,
    cuda_sources=cumsum_source,
    functions=["cumsum_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.
    Optimized with custom CUDA kernel.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumsum_op = cumsum

    def forward(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape), where `*input_shape` 
                              can vary depending on the use case.

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        return self.cumsum_op.cumsum_cuda(x, self.dim)