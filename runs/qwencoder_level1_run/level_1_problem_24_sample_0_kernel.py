import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for online LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void log_softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * dim;
    float* y = output + batch_idx * dim;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;
    
    // First pass: find max
    float thread_max = -INFINITY;
    for (int i = tid; i < dim; i += block_size) {
        thread_max = fmaxf(thread_max, x[i]);
    }
    shared_data[tid] = thread_max;
    __syncthreads();
    
    // Reduction to find max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_data[0];
    __syncthreads();
    
    // Second pass: compute sum of exp(x - max)
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        thread_sum += expf(x[i] - max_val);
    }
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Reduction to compute sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = shared_data[0];
    float log_sum = logf(sum_val);
    __syncthreads();
    
    // Third pass: compute log_softmax
    for (int i = tid; i < dim; i += block_size) {
        y[i] = x[i] - max_val - log_sum;
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int64_t dim) {
    auto batch_size = input.size(0);
    auto feature_dim = input.size(1);
    
    auto output = torch::empty_like(input);
    
    // Use 1D blocks for each sample in the batch
    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = block_size * sizeof(float);
    
    log_softmax_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        feature_dim
    );
    
    return output;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code for LogSoftmax
log_softmax = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a LogSoftmax activation using custom CUDA kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax_func = log_softmax
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return self.log_softmax_func.log_softmax_cuda(x, self.dim)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed