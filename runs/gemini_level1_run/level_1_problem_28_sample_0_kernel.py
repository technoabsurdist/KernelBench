import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and its C++ wrapper for HardSigmoid
hardsigmoid_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for element-wise hardsigmoid
__global__ void hardsigmoid_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val <= -3.0f) {
            out[idx] = 0.0f;
        } else if (val >= 3.0f) {
            out[idx] = 1.0f;
        } else {
            out[idx] = val / 6.0f + 0.5f;
        }
    }
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    auto out = torch::empty_like(x);
    auto size = x.numel();

    if (size == 0) {
        return out;
    }

    // Kernel launch configuration
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    hardsigmoid_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    // Check for any CUDA errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ source for the function signature, required by load_inline
hardsigmoid_cpp_source = "torch::Tensor hardsigmoid_cuda(torch::Tensor x);"

# JIT compile the CUDA code
custom_hardsigmoid_module = load_inline(
    name="custom_hardsigmoid",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_cuda_source,
    functions=["hardsigmoid_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for HardSigmoid.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # The custom module is loaded and its function is assigned for use
        self.custom_hardsigmoid = custom_hardsigmoid_module.hardsigmoid_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the custom HardSigmoid CUDA kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        return self.custom_hardsigmoid(x)

batch_size = 4096
dim = 393216

def get_inputs():
    # Ensure input is on CUDA and is float32 for the custom kernel
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed