import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for reverse cumulative sum
reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer_size,
    const int inner_size,
    const int dim_size
) {
    // Each block handles one "row" of the specified dimension
    int outer_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    if (outer_idx >= outer_size) return;
    
    // Shared memory for reduction
    extern __shared__ char shared_mem_char[];
    scalar_t* shared_mem = reinterpret_cast<scalar_t*>(shared_mem_char);
    
    // Load data into shared memory in reverse order and compute prefix sum
    for (int i = thread_idx; i < dim_size; i += blockDim.x) {
        shared_mem[i] = input[outer_idx * dim_size * inner_size + (dim_size - 1 - i) * inner_size];
    }
    
    __syncthreads();
    
    // Perform inclusive prefix sum (scan) in shared memory
    for (int stride = 1; stride < dim_size; stride *= 2) {
        for (int i = thread_idx; i < dim_size; i += blockDim.x) {
            if (i >= stride) {
                shared_mem[i] += shared_mem[i - stride];
            }
        }
        __syncthreads();
    }
    
    // Write result back in reverse order
    for (int i = thread_idx; i < dim_size; i += blockDim.x) {
        output[outer_idx * dim_size * inner_size + (dim_size - 1 - i) * inner_size] = shared_mem[i];
    }
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int64_t dim) {
    // Ensure input is contiguous
    auto input_contig = input.contiguous();
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_contig));
    
    // Handle negative dimension indexing
    if (dim < 0) dim += input_contig.dim();
    
    auto output = torch::empty_like(input_contig);
    
    if (input_contig.dim() == 1) {
        // Special case for 1D tensor
        dim = 0;
    }
    
    // Calculate dimensions
    int64_t outer_size = 1;
    int64_t inner_size = 1;
    int64_t dim_size = input_contig.size(dim);
    
    for (int i = 0; i < dim; i++) {
        outer_size *= input_contig.size(i);
    }
    for (int i = dim + 1; i < input_contig.dim(); i++) {
        inner_size *= input_contig.size(i);
    }
    
    if (outer_size == 0 || dim_size == 0 || inner_size == 0) {
        return output;
    }
    
    const int block_size = std::min(1024, (int)dim_size);
    const int num_blocks = outer_size;
    const int shared_mem_size = dim_size * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "reverse_cumsum_cuda", ([&] {
        reverse_cumsum_kernel<scalar_t><<<num_blocks, block_size, shared_mem_size>>>(
            input_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            inner_size,
            dim_size
        );
    }));
    
    return output;
}
"""

reverse_cumsum_cpp_source = """
torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code for reverse cumulative sum
reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation along a specified dimension
    using a custom CUDA kernel.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.reverse_cumsum_func = reverse_cumsum

    def forward(self, x):
        return self.reverse_cumsum_func.reverse_cumsum_cuda(x, self.dim)