import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardTanh
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf, fminf

__global__ void hardtanh_kernel(const float* x, float* out, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply HardTanh logic: clamp the value between min_val and max_val
        out[idx] = fmaxf(min_val, fminf(max_val, x[idx]));
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor x) {
    // The original model uses min_val=-1.0 and max_val=1.0
    const float min_val = -1.0f;
    const float max_val = 1.0f;

    auto out = torch::empty_like(x);
    auto size = x.numel();

    // Choose a standard block size
    const int block_size = 1024;
    // Calculate the number of blocks needed to cover all elements
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size,
        min_val,
        max_val
    );

    return out;
}
"""

hardtanh_cpp_source = """
torch::Tensor hardtanh_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code for HardTanh
custom_hardtanh = load_inline(
    name="custom_hardtanh",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a HardTanh activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return custom_hardtanh.hardtanh_cuda(x)