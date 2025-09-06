import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GELU
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Pre-calculated constant for GELU approximation: sqrt(2.0 / PI)
#define M_SQRT_2_OVER_PI 0.7978845608028654f

__global__ void fused_gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        // GELU approximation formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x_cubed = x_val * x_val * x_val;
        float inner_val = M_SQRT_2_OVER_PI * (x_val + 0.044715f * x_cubed);
        out[idx] = 0.5f * x_val * (1.0f + tanhf(inner_val));
    }
}

torch::Tensor gelu_forward_cuda(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    auto out = torch::empty_like(x);
    auto size = x.numel();

    if (size == 0) {
        return out;
    }

    // Kernel launch configuration
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    fused_gelu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size);

    // Check for errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature
gelu_cpp_source = "torch::Tensor gelu_forward_cuda(torch::Tensor x);"

# JIT compile the CUDA kernel
gelu_module = load_inline(
    name="gelu_module",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_cuda_source,
    functions=["gelu_forward_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Implementation of the GELU activation function using a custom fused CUDA kernel.
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled CUDA function
        self.gelu_op = gelu_module.gelu_forward_cuda

    def forward(self, x):
        # Call the custom CUDA kernel
        return self.gelu_op(x)