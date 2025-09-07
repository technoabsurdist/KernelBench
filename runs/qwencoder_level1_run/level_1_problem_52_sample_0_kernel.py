import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmin
argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void argmin_kernel(const float* input, long* output, int outer_size, int reduce_size, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    float min_val = INFINITY;
    long min_idx = 0;
    
    for (int i = 0; i < reduce_size; i++) {
        int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        float val = input[idx];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }
    
    int out_idx = outer_idx * inner_size + inner_idx;
    output[out_idx] = min_idx;
}

torch::Tensor argmin_cuda(torch::Tensor input, int dim) {
    auto sizes = input.sizes();
    int ndim = sizes.size();
    
    // Handle negative dimension
    if (dim < 0) dim += ndim;
    
    int reduce_size = sizes[dim];
    
    // Calculate outer and inner dimensions
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= sizes[i];
    }
    
    // Create output tensor
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < dim; i++) {
        output_sizes.push_back(sizes[i]);
    }
    for (int i = dim + 1; i < ndim; i++) {
        output_sizes.push_back(sizes[i]);
    }
    
    auto output = torch::zeros(output_sizes, torch::kLong).to(input.device());
    
    if (inner_size == 1) {
        // Special case when reducing the last dimension
        const int block_size = 256;
        const int num_blocks = (outer_size + block_size - 1) / block_size;
        
        argmin_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<long>(),
            outer_size,
            reduce_size,
            1
        );
    } else {
        // General case
        dim3 block_size(inner_size < 1024 ? inner_size : 1024);
        dim3 num_blocks(outer_size);
        
        argmin_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<long>(),
            outer_size,
            reduce_size,
            inner_size
        );
    }
    
    return output;
}
"""

argmin_cpp_source = """
torch::Tensor argmin_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for argmin
argmin_cuda = load_inline(
    name="argmin_cuda",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_source,
    functions=["argmin_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return argmin_cuda.argmin_cuda(x, self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [dim]