import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;
    float* shared_max = shared_mem + block_size;
    float* shared_sum = shared_mem + 2 * block_size;
    
    const float* x = input + row * cols;
    float* y = output + row * cols;
    
    // Find maximum value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < cols; i += block_size) {
        thread_max = fmaxf(thread_max, x[i]);
    }
    shared_max[tid] = thread_max;
    __syncthreads();
    
    // Reduce to find row maximum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = shared_max[0];
    __syncthreads();
    
    // Compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float val = expf(x[i] - row_max);
        shared_data[i % block_size] = val;
        thread_sum += val;
        y[i] = val;
    }
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Reduce to find row sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = shared_sum[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < cols; i += block_size) {
        y[i] /= row_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    auto rows = input.size(0);
    auto cols = input.size(1);
    
    const int block_size = 256;
    const int shared_mem_size = 3 * block_size * sizeof(float);
    
    softmax_kernel<<<rows, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, 
        cols
    );
    
    return output;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA Softmax implementation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_func = softmax
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom CUDA Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return self.softmax_func.softmax_cuda(x)