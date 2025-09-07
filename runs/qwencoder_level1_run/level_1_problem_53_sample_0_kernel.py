import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction
min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void min_reduction_kernel(const float* input, float* output, int outer_dim, int reduction_dim, int inner_dim) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_dim || inner_idx >= inner_dim) return;
    
    const float* input_slice = input + outer_idx * reduction_dim * inner_dim + inner_idx;
    float min_val = std::numeric_limits<float>::max();
    
    for (int i = 0; i < reduction_dim; ++i) {
        float val = input_slice[i * inner_dim];
        if (val < min_val) {
            min_val = val;
        }
    }
    
    output[outer_idx * inner_dim + inner_idx] = min_val;
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    auto input_sizes = input.sizes();
    int ndim = input_sizes.size();
    
    // Handle negative dimension indexing
    if (dim < 0) dim += ndim;
    
    // Calculate dimensions
    int outer_dim = 1;
    int reduction_dim = input_sizes[dim];
    int inner_dim = 1;
    
    for (int i = 0; i < dim; ++i) {
        outer_dim *= input_sizes[i];
    }
    for (int i = dim + 1; i < ndim; ++i) {
        inner_dim *= input_sizes[i];
    }
    
    // Create output tensor
    std::vector<int64_t> output_shape;
    for (int i = 0; i < dim; ++i) {
        output_shape.push_back(input_sizes[i]);
    }
    for (int i = dim + 1; i < ndim; ++i) {
        output_shape.push_back(input_sizes[i]);
    }
    
    auto output = torch::zeros(output_shape, torch::kFloat32).to(input.device());
    
    // Launch kernel
    if (inner_dim > 0 && outer_dim > 0) {
        min_reduction_kernel<<<outer_dim, inner_dim>>>(input.data_ptr<float>(), 
                                                       output.data_ptr<float>(), 
                                                       outer_dim, 
                                                       reduction_dim, 
                                                       inner_dim);
    }
    
    return output;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for min reduction
min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return self.min_reduction.min_reduction_cuda(x, self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]  # Reduce over dimension 1