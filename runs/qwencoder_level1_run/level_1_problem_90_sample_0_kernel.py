import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product
cumprod_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void cumprod_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        int offset = row * cols;
        output[offset] = input[offset];
        
        for (int i = 1; i < cols; i++) {
            output[offset + i] = output[offset + i - 1] * input[offset + i];
        }
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input) {
    auto shape = input.sizes();
    int rows = 1;
    int cols = shape[1];
    
    for (int i = 0; i < shape.size() - 1; i++) {
        rows *= shape[i];
    }
    
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int num_blocks = (rows + block_size - 1) / block_size;
    
    cumprod_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    
    return output;
}
"""

cumprod_cpp_source = (
    "torch::Tensor cumprod_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for cumulative product
cumprod = load_inline(
    name="cumprod",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_source,
    functions=["cumprod_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension
    using a custom CUDA kernel for improved performance.
    """

    def __init__(self, dim):
        """
        Initialize the optimized CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod_fn = cumprod

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        # Only optimize for dim=1 case as in the given example
        if self.dim == 1 and x.is_cuda and x.dtype == torch.float32:
            return self.cumprod_fn.cumprod_cuda(x)
        else:
            return torch.cumprod(x, dim=self.dim)