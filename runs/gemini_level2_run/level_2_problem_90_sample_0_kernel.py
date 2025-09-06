import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For erff

// Define constants for the operations
#define LEAKY_RELU_SLOPE 0.2f
#define CLAMP_MIN -1.0f
#define CLAMP_MAX 1.0f
#define M_SQRT1_2f 0.70710678118654752440f // 1/sqrt(2)

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ sum_tensor,
    float* __restrict__ output,
    int total_elements,
    int C,
    int DHW) // D*H*W
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // 1. Load input value
        float val = input[idx];

        // 2. Leaky ReLU
        val = (val > 0) ? val : val * LEAKY_RELU_SLOPE;

        // 3. Add sum_tensor (broadcasted)
        // The input tensor is NCDHW, sum_tensor is C111.
        // We need to find the channel index 'c' for the current element 'idx'.
        int channel_idx = (idx / DHW) % C;
        val += sum_tensor[channel_idx];

        // 4. Clamp
        val = fminf(fmaxf(val, CLAMP_MIN), CLAMP_MAX);

        // 5. GELU (using erff for better precision)
        val = val * 0.5f * (1.0f + erff(val * M_SQRT1_2f));

        // 6. Store final result
        output[idx] = val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor sum_tensor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(sum_tensor.is_cuda(), "Sum tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    // Ensure sum_tensor is contiguous as well, although it's small
    sum_tensor = sum_tensor.contiguous();

    auto output = torch::empty_like(input);
    auto total_elements = input.numel();

    if (total_elements == 0) {
        return output;
    }

    // Get dimensions for broadcasting logic in the kernel
    // Input shape: (N, C, D, H, W)
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    const int DHW = D * H * W;

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_op_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        C,
        DHW
    );

    // Check for any CUDA errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor sum_tensor);"
)

# Compile the inline CUDA code
fused_conv_ops = load_inline(
    name="fused_conv_ops",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, then applies a fused custom CUDA kernel for
    LeakyReLU, summation, clamping, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        # The Conv3d layer is kept as is, since it's highly optimized (cuDNN)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        
        # Store the loaded custom operator
        self.fused_op = fused_conv_ops

    def forward(self, x):
        # 1. Apply the standard, highly optimized Conv3d
        x = self.conv(x)
        
        # 2. Apply the custom fused kernel for the subsequent element-wise operations
        x = self.fused_op.fused_op_cuda(x, self.sum_tensor)
        
        return x