import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    gelu_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    
    return output;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for GELU
gelu_module = load_inline(
    name="gelu_module",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA GELU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_func = gelu_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom CUDA GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return self.gelu_func.gelu_cuda(x)