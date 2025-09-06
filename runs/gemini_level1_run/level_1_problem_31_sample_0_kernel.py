import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for ELU
custom_elu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void elu_kernel(const float* x, float* out, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val > 0.0f) {
            out[idx] = val;
        } else {
            out[idx] = alpha * (expf(val) - 1.0f);
        }
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    // Ensure the input tensor is a contiguous CUDA tensor of type float
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be a float32 tensor");

    auto out = torch::empty_like(x);
    const int size = x.numel();

    if (size == 0) {
        return out;
    }

    // Use 1024 threads per block, a common choice for element-wise kernels
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        size
    );
    
    // Check for errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# Define the C++ function signature
custom_elu_cpp_source = """
torch::Tensor elu_cuda(torch::Tensor x, float alpha);
"""

# Compile the inline CUDA code
custom_elu = load_inline(
    name="custom_elu",
    cpp_sources=custom_elu_cpp_source,
    cuda_sources=custom_elu_source,
    functions=["elu_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs an ELU activation using a custom CUDA kernel.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch::Tensor: Output tensor with ELU applied, same shape as input.
        """
        # The custom kernel requires a contiguous CUDA tensor.
        # This ensures the model works correctly even with CPU or non-contiguous inputs.
        x_cuda_contiguous = x.contiguous().cuda()
        return custom_elu.elu_cuda(x_cuda_contiguous, self.alpha)