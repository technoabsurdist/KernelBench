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
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* input_row = input + batch_idx * dim;
    float* output_row = output + batch_idx * dim;
    float norm = norms[batch_idx];
    
    for (int i = tid; i < dim; i += block_size) {
        output_row[i] = input_row[i] / norm;
    }
}

__global__ void squared_sum_kernel(const float* input, float* squared_sums, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* input_row = input + batch_idx * dim;
    float sum = 0.0f;
    
    for (int i = tid; i < dim; i += block_size) {
        float val = input_row[i];
        sum += val * val;
    }
    
    // Reduction within block
    typedef cub::BlockReduce<float, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float aggregate = BlockReduce(temp_storage).Sum(sum);
    
    if (tid == 0) {
        squared_sums[batch_idx] = aggregate;
    }
}

torch::Tensor l2_normalize_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int dim = input.size(1);
    
    // Create output tensor
    auto output = torch::zeros_like(input);
    
    // Compute squared sums for each row
    auto squared_sums = torch::zeros({batch_size}, torch::kFloat32);
    
    // Launch kernel to compute squared sums
    squared_sum_kernel<<<batch_size, 1024>>>(
        input.data_ptr<float>(),
        squared_sums.data_ptr<float>(),
        batch_size,
        dim
    );
    
    // Compute L2 norms
    auto norms = torch::sqrt(squared_sums + 1e-12);
    
    // Launch kernel to normalize
    l2_norm_kernel<<<batch_size, 1024>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        norms.data_ptr<float>(),
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
    def __init__(self) -> None:
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
dim = 65535

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []