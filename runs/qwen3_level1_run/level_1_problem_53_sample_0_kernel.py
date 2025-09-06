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
    int inner_idx = blockIdx.y;
    
    if (outer_idx >= outer_dim || inner_idx >= inner_dim) return;
    
    const float* row = input + outer_idx * reduction_dim * inner_dim + inner_idx;
    float min_val = std::numeric_limits<float>::max();
    
    for (int i = threadIdx.x; i < reduction_dim; i += blockDim.x) {
        float val = row[i * inner_dim];
        if (val < min_val) min_val = val;
    }
    
    // Reduction within block
    __shared__ float shared_min[32]; // Assuming max 32 threads per block for simplicity
    int tid = threadIdx.x;
    shared_min[tid] = min_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_min[tid + stride] < shared_min[tid]) {
                shared_min[tid] = shared_min[tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[outer_idx * inner_dim + inner_idx] = shared_min[0];
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    auto input_sizes = input.sizes();
    auto output_sizes = input_sizes.vec();
    output_sizes.erase(output_sizes.begin() + dim);
    
    auto output = torch::zeros(output_sizes, input.options());
    
    int outer_dim = 1;
    for (int i = 0; i < dim; i++) {
        outer_dim *= input_sizes[i];
    }
    
    int reduction_dim = input_sizes[dim];
    
    int inner_dim = 1;
    for (int i = dim + 1; i < input_sizes.size(); i++) {
        inner_dim *= input_sizes[i];
    }
    
    const int block_size = 32;
    dim3 grid_size(outer_dim, inner_dim);
    
    min_reduction_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        outer_dim, 
        reduction_dim, 
        inner_dim
    );
    
    return output;
}
"""

min_reduction_cpp_source = (
    "torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);"
)

# Compile the inline CUDA code for min reduction
min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for min reduction.
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
        Applies min reduction over the specified dimension to the input tensor using custom CUDA kernel.

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