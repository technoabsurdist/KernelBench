import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: multiply -> leaky_relu -> gelu
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU approximation, same as in PyTorch
__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3.0f))));
}

// Leaky ReLU with default negative_slope=0.01
__device__ __forceinline__ float leaky_relu(float x) {
    const float negative_slope = 0.01f;
    return (x > 0.0f) ? x : negative_slope * x;
}

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ multiplier,
    float* __restrict__ output,
    int total_elements,
    int channels,
    int plane_size) {

    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        // Calculate channel index for broadcasting the multiplier
        // idx = n * C*H*W + c * H*W + h * W + w
        // idx / plane_size = n * C + c
        // (idx / plane_size) % C = c
        int c = (idx / plane_size) % channels;

        // Read input and multiplier
        float input_val = input[idx];
        float multiplier_val = multiplier[c];

        // Fused operations: multiply -> leaky_relu -> gelu
        float result = input_val * multiplier_val;
        result = leaky_relu(result);
        result = gelu_approx(result);

        // Write output
        output[idx] = result;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor multiplier) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(multiplier.is_cuda(), "Multiplier must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(multiplier.is_contiguous(), "Multiplier must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(multiplier.scalar_type() == torch::kFloat32, "Multiplier must be a float32 tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(multiplier.dim() == 3, "Multiplier must be a 3D tensor");
    TORCH_CHECK(input.size(1) == multiplier.size(0), "Input channels must match multiplier size");

    auto output = torch::empty_like(input);
    const int total_elements = input.numel();
    if (total_elements == 0) {
        return output;
    }

    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int plane_size = height * width;

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_op_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        channels,
        plane_size
    );

    // Check for any CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor multiplier);"
)

# JIT compile the inline CUDA code
# This will be compiled only once when the Python module is first imported.
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, followed by a custom fused operation that
    combines multiplication by a learnable scalar, LeakyReLU, and GELU into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        # The convolution layer remains a standard PyTorch operator, as it's highly optimized.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # The learnable multiplier parameter is the same as in the original model.
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        
        # The custom fused operator is loaded and assigned here.
        # The original LeakyReLU and the functional.gelu call are replaced by this.
        self.fused_op = fused_op.fused_op_cuda

    def forward(self, x):
        # 1. Apply the standard, optimized convolution.
        x = self.conv(x)
        
        # 2. Apply the custom fused kernel for the sequence of element-wise operations.
        # This avoids intermediate tensor allocations and reduces kernel launch overhead.
        x = self.fused_op(x, self.multiplier)
        
        return x