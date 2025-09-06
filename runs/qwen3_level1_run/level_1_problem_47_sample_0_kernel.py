import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum reduction
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void sum_reduction_kernel(const float* input, float* output, int outer_dim, int reduce_dim, int inner_dim) {
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y;
    
    if (outer_idx >= outer_dim || inner_idx >= inner_dim) return;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        sum += input[outer_idx * reduce_dim * inner_dim + i * inner_dim + inner_idx];
    }
    
    // Use shared memory for reduction within block
    __shared__ float shared_data[1024]; // Assuming max block size of 1024
    shared_data[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (threadIdx.x == 0) {
        output[outer_idx * inner_dim + inner_idx] = shared_data[0];
    }
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim) {
    auto input_sizes = input.sizes();
    auto output_sizes = input_sizes.vec();
    output_sizes[dim] = 1;
    
    auto output = torch::zeros(output_sizes, input.options());
    
    int outer_dims = 1;
    int reduce_dim = input_sizes[dim];
    int inner_dims = 1;
    
    for (int i = 0; i < dim; i++) {
        outer_dims *= input_sizes[i];
    }
    for (int i = dim + 1; i < input_sizes.size(); i++) {
        inner_dims *= input_sizes[i];
    }
    
    dim3 grid(outer_dims, inner_dims);
    dim3 block(min(reduce_dim, 1024));
    
    sum_reduction_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        outer_dims, 
        reduce_dim, 
        inner_dims
    );
    
    return output;
}
"""

sum_reduction_cpp_source = """
torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for sum reduction
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return self.sum_reduction.sum_reduction_cuda(x, self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [reduce_dim]