import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void l2_norm_kernel(const float* input, float* output, const float* norms, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * dim;
    
    if (idx < total_elements) {
        int batch_idx = idx / dim;
        output[idx] = input[idx] / norms[batch_idx];
    }
}

__global__ void squared_sum_kernel(const float* input, float* squared_sums, int batch_size, int dim) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            float val = input[batch_idx * dim + i];
            sum += val * val;
        }
        squared_sums[batch_idx] = sum;
    }
}

__global__ void sqrt_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = sqrtf(data[idx]);
    }
}

torch::Tensor l2_normalize_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    
    // Create output tensor
    auto output = torch::zeros_like(input);
    
    // Compute squared sums for each batch
    auto squared_sums = torch::zeros({batch_size}, torch::kFloat32);
    
    // Launch kernel to compute squared sums
    const int block_size = 256;
    const int num_blocks_sum = (batch_size + block_size - 1) / block_size;
    
    squared_sum_kernel<<<num_blocks_sum, block_size>>>(
        input.data_ptr<float>(), 
        squared_sums.data_ptr<float>(), 
        batch_size, 
        dim
    );
    
    // Take square root to get L2 norms
    const int num_blocks_norm = (batch_size + block_size - 1) / block_size;
    sqrt_kernel<<<num_blocks_norm, block_size>>>(
        squared_sums.data_ptr<float>(), 
        batch_size
    );
    
    // Normalize input by L2 norms
    const int total_elements = batch_size * dim;
    const int num_blocks_normalize = (total_elements + block_size - 1) / block_size;
    
    l2_norm_kernel<<<num_blocks_normalize, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        squared_sums.data_ptr<float>(), 
        batch_size, 
        dim
    );
    
    return output;
}
"""

l2_norm_cpp_source = (
    "torch::Tensor l2_normalize_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for L2 normalization
l2_normalize = load_inline(
    name="l2_normalize",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_normalize_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization with custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L2Norm layer.
        """
        super(ModelNew, self).__init__()
        self.l2_normalize = l2_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return self.l2_normalize.l2_normalize_cuda(x)

batch_size = 32768
# choose dim so total <2^31
dim = 65535

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []