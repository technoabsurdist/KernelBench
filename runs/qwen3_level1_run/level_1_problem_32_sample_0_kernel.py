import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardTanh activation
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void hardtanh_kernel(const float* input, float* output, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        if (val < min_val) {
            output[idx] = min_val;
        } else if (val > max_val) {
            output[idx] = max_val;
        } else {
            output[idx] = val;
        }
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    auto output = torch::empty_like(input);
    auto size = input.numel();
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    hardtanh_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, 
        min_val, 
        max_val
    );
    
    return output;
}
"""

hardtanh_cpp_source = (
    "torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val);"
)

# Compile the inline CUDA code for HardTanh
hardtanh = load_inline(
    name="hardtanh",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for HardTanh activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.min_val = -1.0
        self.max_val = 1.0
        self.hardtanh_func = hardtanh
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized HardTanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return self.hardtanh_func.hardtanh_cuda(x, self.min_val, self.max_val)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed