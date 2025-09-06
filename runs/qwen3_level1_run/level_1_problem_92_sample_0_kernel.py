import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for exclusive cumulative sum
exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void exclusive_cumsum_kernel(const float* input, float* output, int dim_size, int stride, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || dim_idx >= dim_size) return;
    
    int idx = batch_idx * stride + dim_idx;
    
    if (dim_idx == 0) {
        output[idx] = 0.0f;
    } else {
        output[idx] = output[idx - 1] + input[idx - 1];
    }
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int dim) {
    auto output = torch::zeros_like(input);
    int dim_size = input.size(dim);
    int batch_size = 1;
    
    for (int i = 0; i < dim; i++) {
        batch_size *= input.size(i);
    }
    
    int stride = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        stride *= input.size(i);
    }
    
    if (dim == input.dim() - 1) {
        const int block_size_x = 256;
        const int block_size_y = 1;
        const dim3 block_size(block_size_x, block_size_y);
        const dim3 grid_size((batch_size + block_size_x - 1) / block_size_x, dim_size);
        
        exclusive_cumsum_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            dim_size, 
            stride, 
            batch_size
        );
    } else {
        const int block_size = 256;
        const int num_blocks = (batch_size + block_size - 1) / block_size;
        
        // For non-last dimensions, we need a different approach
        // This is a simplified version that works for the given use case
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < stride; j++) {
                int base_idx = i * dim_size * stride + j;
                output[base_idx] = 0.0f;
                for (int k = 1; k < dim_size; k++) {
                    int out_idx = base_idx + k * stride;
                    int in_idx = base_idx + (k - 1) * stride;
                    output[out_idx] = output[in_idx] + input[in_idx];
                }
            }
        }
    }
    
    return output;
}
"""

exclusive_cumsum_cpp_source = """
torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for exclusive cumulative sum
exclusive_cumsum = load_inline(
    name="exclusive_cumsum",
    cpp_sources=exclusive_cumsum_cpp_source,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).
    Optimized with custom CUDA kernel.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.exclusive_cumsum = exclusive_cumsum

    def forward(self, x):
        return self.exclusive_cumsum.exclusive_cumsum_cuda(x, self.dim)