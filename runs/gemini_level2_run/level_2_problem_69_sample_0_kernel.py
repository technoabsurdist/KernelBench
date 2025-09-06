import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused HardSwish + ReLU
# The fused operation is f(x) = relu(hardswish(x)).
#
# Analysis of f(x):
# - If x < 0:
#   - If x <= -3, hardswish(x) = 0, so f(x) = relu(0) = 0.
#   - If -3 < x < 0, hardswish(x) = x * (x+3) / 6, which is negative. So f(x) = relu(negative) = 0.
#   - Thus, for all x < 0, f(x) = 0.
# - If x >= 0:
#   - hardswish(x) is always non-negative, so f(x) = relu(hardswish(x)) = hardswish(x).
#   - If 0 <= x < 3, hardswish(x) = x * (x + 3) / 6.
#   - If x >= 3, hardswish(x) = x.
#
# Final logic for the fused kernel:
# - if x < 0: 0
# - if 0 <= x < 3: x * (x + 3) / 6
# - if x >= 3: x
fused_hardswish_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_hardswish_relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        if (val < 0.0f) {
            output[idx] = 0.0f;
        } else if (val < 3.0f) {
            output[idx] = val * (val + 3.0f) / 6.0f;
        } else {
            output[idx] = val;
        }
    }
}

torch::Tensor fused_hardswish_relu_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto out = torch::empty_like(input);
    auto size = input.numel();

    if (size == 0) {
        return out;
    }

    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_hardswish_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

fused_hardswish_relu_cpp_source = (
    "torch::Tensor fused_hardswish_relu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for the fused operation
fused_op = load_inline(
    name="fused_hardswish_relu",
    cpp_sources=fused_hardswish_relu_cpp_source,
    cuda_sources=fused_hardswish_relu_source,
    functions=["fused_hardswish_relu_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution and then applies a custom fused
    HardSwish + ReLU activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Assign the compiled custom CUDA function
        self.fused_activation = fused_op.fused_hardswish_relu_cuda

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        # Apply the fused operation using the custom kernel
        x = self.fused_activation(x)
        return x