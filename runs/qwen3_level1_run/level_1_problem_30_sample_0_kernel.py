import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softsign activation
softsign_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void softsign_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x / (1.0f + fabsf(x));
    }
}

torch::Tensor softsign_cuda(torch::Tensor input) {
    const auto input_size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int grid_size = (input_size + block_size - 1) / block_size;
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    softsign_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        input_size
    );
    
    return output;
}
"""

softsign_cpp_source = """
#include <torch/extension.h>
torch::Tensor softsign_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code for Softsign
softsign_module = load_inline(
    name="softsign_cuda",
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_source,
    functions=["softsign_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softsign activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softsign_func = softsign_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softsign activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softsign applied, same shape as input.
        """
        return self.softsign_func.softsign_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed