import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus activation
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void softplus_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        if (x > 20.0f) {
            output[idx] = x;  // For large x, softplus(x) ≈ x
        } else if (x < -20.0f) {
            output[idx] = expf(x);  // For very negative x, softplus(x) ≈ exp(x)
        } else {
            output[idx] = log1pf(expf(x));  // Standard softplus computation
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    softplus_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    
    return output;
}
"""

softplus_cpp_source = (
    "torch::Tensor softplus_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Softplus activation
softplus_module = load_inline(
    name="softplus_cuda",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus_func = softplus_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Softplus activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return self.softplus_func.softplus_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed