import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Sigmoid activation
sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void sigmoid_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

sigmoid_cpp_source = (
    "torch::Tensor sigmoid_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Sigmoid activation
sigmoid = load_inline(
    name="sigmoid",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_source,
    functions=["sigmoid_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Sigmoid activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_op = sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return self.sigmoid_op.sigmoid_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed