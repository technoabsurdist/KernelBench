import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ReLU + HardSwish
relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void relu_hardswish_fused_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // First operation: ReLU
        float relu_val = fmaxf(0.0f, input[idx]);
        
        // Second operation: HardSwish applied to the result of ReLU
        // The original python is: x * torch.clamp((x + 3) / 6, 0, 1)
        // Here, 'x' is the result of the previous ReLU operation.
        float temp = (relu_val + 3.0f) / 6.0f;
        float clamped_val = fminf(fmaxf(temp, 0.0f), 1.0f);
        
        output[idx] = relu_val * clamped_val;
    }
}

torch::Tensor relu_hardswish_fused_cuda(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    // Create an output tensor
    auto output = torch::empty_like(input);
    auto size = input.numel();

    if (size == 0) {
        return output;
    }

    // Kernel launch configuration
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    relu_hardswish_fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

# Define the C++ source for the function signature
relu_hardswish_cpp_source = """
torch::Tensor relu_hardswish_fused_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code for the fused activation
fused_activation = load_inline(
    name="relu_hardswish_fused",
    cpp_sources=relu_hardswish_cpp_source,
    cuda_sources=relu_hardswish_source,
    functions=["relu_hardswish_fused_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses ReLU and HardSwish into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Store the compiled function
        self.fused_activation_op = fused_activation

    def forward(self, x):
        x = self.conv(x)
        # Apply the custom fused activation
        x = self.fused_activation_op.relu_hardswish_fused_cuda(x)
        return x