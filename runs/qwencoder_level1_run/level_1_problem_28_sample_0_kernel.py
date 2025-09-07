import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void hardsigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        if (x <= -3.0f) {
            output[idx] = 0.0f;
        } else if (x >= 3.0f) {
            output[idx] = 1.0f;
        } else {
            output[idx] = (x + 3.0f) / 6.0f;
        }
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    auto size = input.numel();
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    hardsigmoid_kernel<<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    
    return output;
}
"""

hardsigmoid_cpp_source = """
torch::Tensor hardsigmoid_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code for HardSigmoid
hardsigmoid = load_inline(
    name="hardsigmoid",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a HardSigmoid activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardsigmoid_fn = hardsigmoid
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        return self.hardsigmoid_fn.hardsigmoid_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed