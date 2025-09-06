import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LeakyReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* x, float* out, int size, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto out = torch::empty_like(x);
    auto size = x.numel();

    if (size == 0) {
        return out;
    }

    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    leaky_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size,
        negative_slope
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

leaky_relu_cpp_source = (
    "torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope);"
)

# Compile the inline CUDA code for LeakyReLU
leaky_relu_cuda_module = load_inline(
    name="leaky_relu_cuda_module",
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_source,
    functions=["leaky_relu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation using a custom CUDA kernel.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        # The custom kernel expects a contiguous tensor
        return leaky_relu_cuda_module.leaky_relu_cuda(x.contiguous(), self.negative_slope)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed