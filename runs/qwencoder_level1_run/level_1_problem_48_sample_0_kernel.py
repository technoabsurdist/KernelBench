import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for mean reduction
mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void mean_reduction_kernel(const float* input, float* output, 
                                     int outer_size, int reduction_size, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    float sum = 0.0f;
    for (int i = 0; i < reduction_size; ++i) {
        int input_idx = outer_idx * reduction_size * inner_size + i * inner_size + inner_idx;
        sum += input[input_idx];
    }
    
    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = sum / static_cast<float>(reduction_size);
}

torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    auto input_sizes = input.sizes();
    auto output_sizes = input_sizes.vec();
    output_sizes.erase(output_sizes.begin() + dim);
    
    auto output = torch::zeros(output_sizes, input.options());
    
    // Set up dimensions for kernel
    int outer_size = 1;
    int reduction_size = input_sizes[dim];
    int inner_size = 1;
    
    for (int i = 0; i < dim; ++i) {
        outer_size *= input_sizes[i];
    }
    for (int i = dim + 1; i < input_sizes.size(); ++i) {
        inner_size *= input_sizes[i];
    }
    
    if (inner_size == 0) inner_size = 1;
    
    const int block_size = min(1024, inner_size);
    const int num_blocks = outer_size;
    
    // Ensure we're on the right device
    at::cuda::CUDAGuard device_guard(input.device());
    
    mean_reduction_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        outer_size,
        reduction_size,
        inner_size
    );
    
    return output;
}
"""

mean_reduction_cpp_source = """
torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for mean reduction
mean_reduction = load_inline(
    name="mean_reduction",
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for mean reduction.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)