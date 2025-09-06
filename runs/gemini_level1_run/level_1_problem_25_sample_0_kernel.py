import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused Swish operation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For expf

__global__ void swish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Fused operation: out = x * sigmoid(x)
        // sigmoid(x) = 1.0f / (1.0f + exp(-x))
        float val = x[idx];
        out[idx] = val / (1.0f + expf(-val));
    }
}

torch::Tensor swish_forward_cuda(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    // Allocate output tensor
    auto out = torch::empty_like(x);
    int size = x.numel();

    // Kernel launch configuration
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    swish_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}
"""

swish_cpp_source = "torch::Tensor swish_forward_cuda(torch::Tensor x);"

# Compile the inline CUDA code
swish_op = load_inline(
    name="swish_op",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Swish activation using a custom fused CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        return swish_op.swish_forward_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed