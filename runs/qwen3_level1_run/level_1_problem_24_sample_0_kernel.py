import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for online LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void log_softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Shared memory for reduction operations
    extern __shared__ float shared_mem[];
    float* max_vals = shared_mem;           // Shared memory for max values
    float* sum_vals = &shared_mem[block_size]; // Shared memory for sum values
    
    const float* input_row = input + row * cols;
    float* output_row = output + row * cols;
    
    // Step 1: Find maximum value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < cols; i += block_size) {
        thread_max = fmaxf(thread_max, input_row[i]);
    }
    max_vals[tid] = thread_max;
    __syncthreads();
    
    // Reduction to find maximum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + stride]);
        }
        __syncthreads();
    }
    
    float row_max = max_vals[0];
    __syncthreads();
    
    // Step 2: Compute sum of exponentials
    float thread_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        thread_sum += expf(input_row[i] - row_max);
    }
    sum_vals[tid] = thread_sum;
    __syncthreads();
    
    // Reduction to compute sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_vals[tid] += sum_vals[tid + stride];
        }
        __syncthreads();
    }
    
    float row_sum = sum_vals[0];
    float log_sum = logf(row_sum);
    __syncthreads();
    
    // Step 3: Compute log softmax values
    for (int i = tid; i < cols; i += block_size) {
        output_row[i] = input_row[i] - row_max - log_sum;
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int64_t dim) {
    // Ensure we're on the correct device
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto output = torch::empty_like(input);
    
    if (dim == -1) dim = input.dim() - 1;
    
    // For simplicity, assume dim=-1 or dim=1 (last dimension)
    if (dim != input.dim() - 1) {
        AT_ERROR("Only last dimension supported for this implementation");
    }
    
    int rows = 1;
    for (int i = 0; i < dim; i++) {
        rows *= input_sizes[i];
    }
    int cols = input_sizes[dim];
    
    // Limit block size to 1024 threads
    const int block_size = min(1024, max(32, cols));
    const int shared_mem_size = 2 * block_size * sizeof(float); // For max_vals and sum_vals
    
    // Launch kernel
    log_softmax_kernel<<<rows, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, 
        cols
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

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed

# Keep the same parameters as the original model
batch_size = 4096
dim = 393216