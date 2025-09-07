import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for masked cumulative sum
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void masked_cumsum_kernel(
    const float* x,
    const bool* mask,
    float* output,
    int batch_size,
    int seq_len
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for partial sums and valid flags
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;
    bool* shared_mask = (bool*)&shared_data[block_size];
    float* shared_prefix = (float*)&shared_mask[block_size];
    
    float cumulative_sum = 0.0f;
    
    // Process elements in chunks
    for (int i = tid; i < seq_len; i += block_size) {
        int idx = batch_idx * seq_len + i;
        float val = mask[idx] ? x[idx] : 0.0f;
        
        shared_data[tid] = val;
        shared_mask[tid] = mask[idx];
        __syncthreads();
        
        // Perform inclusive scan on shared memory
        for (int stride = 1; stride < min(block_size, seq_len - (i/block_size)*block_size); stride *= 2) {
            float temp_val = 0.0f;
            if (tid >= stride) {
                temp_val = shared_data[tid - stride];
            }
            __syncthreads();
            if (tid >= stride) {
                shared_data[tid] += temp_val;
            }
            __syncthreads();
        }
        
        // Add cumulative sum from previous chunks
        shared_prefix[tid] = cumulative_sum;
        __syncthreads();
        
        // Handle cross-chunk dependencies
        if (tid == block_size - 1) {
            cumulative_sum += shared_data[tid];
        }
        __syncthreads();
        
        // Write result
        if (i < seq_len) {
            output[idx] = shared_data[tid] + shared_prefix[tid];
        }
        
        __syncthreads();
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto output = torch::zeros_like(x);
    
    if (dim != -1 && dim != x.dim() - 1) {
        // For non-last dimension, we need to transpose
        std::vector<int64_t> dims(x.dim());
        std::iota(dims.begin(), dims.end(), 0);
        dims[dim] = x.dim() - 1;
        dims[x.dim() - 1] = dim;
        
        x = x.permute(dims);
        mask = mask.permute(dims);
        output = output.permute(dims);
    }
    
    auto sizes = x.sizes();
    int batch_size = 1;
    for (int i = 0; i < x.dim() - 1; i++) {
        batch_size *= sizes[i];
    }
    int seq_len = sizes[x.dim() - 1];
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = block_size * (sizeof(float) * 2 + sizeof(bool));
    
    masked_cumsum_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len
    );
    
    if (dim != -1 && dim != x.dim() - 1) {
        // Transpose back
        std::vector<int64_t> dims(x.dim());
        std::iota(dims.begin(), dims.end(), 0);
        dims[dim] = x.dim() - 1;
        dims[x.dim() - 1] = dim;
        
        output = output.permute(dims);
    }
    
    return output;
}
"""

masked_cumsum_cpp_source = """
torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim);
"""

# Compile the inline CUDA code for masked cumulative sum
masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.
    Optimized with custom CUDA kernel.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)