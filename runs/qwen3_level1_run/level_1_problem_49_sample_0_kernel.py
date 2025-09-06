import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for max reduction
max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void max_reduction_kernel(const float* input, float* output, int reduction_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < reduction_size; ++i) {
            float val = input[idx * reduction_size + i];
            max_val = fmaxf(max_val, val);
        }
        output[idx] = max_val;
    }
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int dim) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto input_dim = input.dim();
    
    // Handle negative dimension indexing
    if (dim < 0) dim += input_dim;
    
    int reduction_size = input_sizes[dim];
    int output_size = 1;
    
    // Calculate output size
    for (int i = 0; i < input_dim; ++i) {
        if (i != dim) {
            output_size *= input_sizes[i];
        }
    }
    
    // Create output tensor
    std::vector<int64_t> output_shape;
    for (int i = 0; i < input_dim; ++i) {
        if (i != dim) {
            output_shape.push_back(input_sizes[i]);
        }
    }
    
    auto output = torch::empty(output_shape, input.options());
    
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    max_reduction_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        reduction_size, 
        output_size
    );
    
    return output;
}
"""

max_reduction_cpp_source = """
torch::Tensor max_reduction_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for max reduction
max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduction = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return self.max_reduction.max_reduction_cuda(x, self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]  # Reduce over dimension 1