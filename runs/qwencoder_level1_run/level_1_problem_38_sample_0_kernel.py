import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void l1_norm_kernel(const float* x, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    const float* x_row = x + batch_idx * dim;
    float* out_row = output + batch_idx * dim;
    
    // Compute sum of absolute values
    float sum = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        sum += fabsf(x_row[i]);
    }
    
    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Broadcast the mean
    float mean = sdata[0] / dim;
    if (mean == 0.0f) mean = 1.0f; // Avoid division by zero
    
    // Normalize
    for (int i = tid; i < dim; i += block_size) {
        out_row[i] = x_row[i] / mean;
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto output = torch::zeros_like(x);
    
    // Use 1D blocks for better occupancy
    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = block_size * sizeof(float);
    
    l1_norm_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        dim
    );
    
    return output;
}
"""

l1_norm_cpp_source = (
    "torch::Tensor l1_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for L1 normalization
l1_norm = load_inline(
    name="l1_norm",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization with custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return self.l1_norm.l1_norm_cuda(x)

batch_size = 32768
dim = 65535

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []