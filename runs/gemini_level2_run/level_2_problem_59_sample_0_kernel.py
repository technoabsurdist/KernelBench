import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Swish activation and scaling
fused_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_swish_scale_kernel(const float* input, float* output, float scaling_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float val = input[idx];
        // Swish activation: x * sigmoid(x)
        const float swish_val = val * (1.0f / (1.0f + expf(-val)));
        // Apply scaling
        output[idx] = swish_val * scaling_factor;
    }
}

torch::Tensor fused_swish_scale_cuda(torch::Tensor input, float scaling_factor) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    // Allocate output tensor
    auto output = torch::empty_like(input);
    const auto size = input.numel();

    // Kernel launch configuration
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    fused_swish_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        size
    );

    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for the function signature
fused_swish_scale_cpp_source = """
torch::Tensor fused_swish_scale_cuda(torch::Tensor input, float scaling_factor);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_swish_scale_cpp_source,
    cuda_sources=fused_swish_scale_source,
    functions=["fused_swish_scale_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel to fuse Swish activation and scaling.
    The matrix multiplication is kept in PyTorch to leverage highly optimized cuBLAS libraries.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_swish_scale = fused_op.fused_swish_scale_cuda

    def forward(self, x):
        # 1. Perform matrix multiplication using the efficient nn.Linear
        x = self.matmul(x)
        # 2. Apply the fused Swish and scaling operation using our custom kernel
        x = self.fused_swish_scale(x, self.scaling_factor)
        return x