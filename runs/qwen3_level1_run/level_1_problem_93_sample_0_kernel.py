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
    
    float running_sum = 0.0f;
    
    // Process elements in chunks
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += block_size) {
        int idx = chunk_start + tid;
        int shared_idx = tid;
        
        // Load data into shared memory
        if (idx < seq_len) {
            shared_data[shared_idx] = x[batch_idx * seq_len + idx];
            shared_mask[shared_idx] = mask[batch_idx * seq_len + idx];
        } else {
            shared_data[shared_idx] = 0.0f;
            shared_mask[shared_idx] = false;
        }
        
        __syncthreads();
        
        // Perform inclusive scan with mask
        for (int stride = 1; stride < min(block_size, seq_len - chunk_start); stride *= 2) {
            float temp_data = 0.0f;
            bool temp_mask = false;
            
            if (tid >= stride) {
                temp_data = shared_data[shared_idx - stride];
                temp_mask = shared_mask[shared_idx - stride];
            }
            
            __syncthreads();
            
            if (tid >= stride && shared_mask[shared_idx]) {
                if (temp_mask) {
                    shared_data[shared_idx] += temp_data;
                }
            }
            
            __syncthreads();
        }
        
        // Add running sum from previous chunks
        if (shared_mask[shared_idx]) {
            shared_data[shared_idx] += running_sum;
        }
        
        // Update running sum for next chunk
        if (tid == min(block_size, seq_len - chunk_start) - 1) {
            shared_prefix[0] = shared_mask[shared_idx] ? shared_data[shared_idx] : running_sum;
        }
        
        __syncthreads();
        
        if (tid == 0) {
            running_sum = shared_prefix[0];
        }
        
        __syncthreads();
        
        // Write results back to global memory
        if (idx < seq_len) {
            output[batch_idx * seq_len + idx] = shared_data[shared_idx];
        }
        
        __syncthreads();
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim) {
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    
    auto output = torch::zeros_like(x);
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    
    if (dim != -1 && dim != x.dim() - 1) {
        // For non-last dimension, we need to transpose
        std::vector<int64_t> permute_order(x.dim());
        std::iota(permute_order.begin(), permute_order.end(), 0);
        permute_order[dim] = x.dim() - 1;
        permute_order[x.dim() - 1] = dim;
        
        auto x_transposed = x.permute(permute_order);
        auto mask_transposed = mask.permute(permute_order);
        auto output_transposed = masked_cumsum_cuda(x_transposed, mask_transposed, -1);
        return output_transposed.permute(permute_order);
    }
    
    const int block_size = 512;
    const int num_blocks = batch_size;
    const int shared_mem_size = block_size * (sizeof(float) * 2 + sizeof(bool));
    
    masked_cumsum_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len
    );
    
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
    Optimized model that performs a masked cumulative sum using custom CUDA kernel.
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