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
        float sum = 0.0f;
        for (int i = 0; i < dim_idx; i++) {
            sum += input[batch_idx * stride + i];
        }
        output[idx] = sum;
    }
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int dim) {
    auto output = torch::zeros_like(input);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    if (dim != -1 && dim != input.dim() - 1) {
        // For simplicity, we only optimize for the last dimension
        // Fall back to PyTorch implementation for other dimensions
        auto exclusive_cumsum = torch::cat({
            torch::zeros_like(input.select(dim, 0).unsqueeze(dim)),
            input
        }, dim).slice(dim, 0, input.size(dim));
        return torch::cumsum(exclusive_cumsum, dim);
    }
    
    int dim_size = input.size(-1);
    int batch_size = 1;
    for (int i = 0; i < input.dim() - 1; i++) {
        batch_size *= input.size(i);
    }
    int stride = dim_size;
    
    dim3 block_size(16, 16);
    dim3 num_blocks((batch_size + block_size.x - 1) / block_size.x,
                    (dim_size + block_size.y - 1) / block_size.y);
    
    exclusive_cumsum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_size,
        stride,
        batch_size
    );
    
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

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]