import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation (min + bias + scale)
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fminf

__global__ void fused_op_kernel(
    const float* __restrict__ conv_out,
    float* __restrict__ out,
    const float* __restrict__ bias,
    const float constant_value,
    const float scaling_factor,
    const long total_elements,
    const int C,
    const int H,
    const int W) {

    const long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = H * W;

    if (idx < total_elements) {
        // Calculate channel index for bias broadcasting.
        // The memory layout is NCHW.
        // idx = n * C * H * W + c * H * W + h * W + w
        // idx / (H * W) = n * C + c
        // (idx / (H * W)) % C = c
        const int channel_idx = (idx / HW) % C;

        // Fused operation: min, add bias, scale
        float val = conv_out[idx];
        val = fminf(val, constant_value);
        val = val + bias[channel_idx];
        val = val * scaling_factor;
        out[idx] = val;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor conv_out,
    torch::Tensor bias,
    float constant_value,
    float scaling_factor) {

    // Input validation
    TORCH_CHECK(conv_out.is_cuda(), "conv_out must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(conv_out.scalar_type() == torch::kFloat32, "conv_out must be a float32 tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be a float32 tensor");

    // Ensure tensors are contiguous for correct memory access
    conv_out = conv_out.contiguous();
    bias = bias.contiguous();

    const auto N = conv_out.size(0);
    const auto C = conv_out.size(1);
    const auto H = conv_out.size(2);
    const auto W = conv_out.size(3);
    const long total_elements = conv_out.numel();

    // Check bias shape for broadcasting
    TORCH_CHECK(bias.dim() == 3 && bias.size(0) == C && bias.size(1) == 1 && bias.size(2) == 1, "Bias must have shape (C, 1, 1)");

    auto out = torch::empty_like(conv_out);

    // Kernel launch configuration
    const int block_size = 1024;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_op_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        out.data_ptr<float>(),
        bias.data_ptr<float>(),
        constant_value,
        scaling_factor,
        total_elements,
        C, H, W);
    
    // Check for any CUDA errors that may have occurred during the kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor conv_out,
    torch::Tensor bias,
    float constant_value,
    float scaling_factor);
"""

# JIT compile the CUDA and C++ code
fused_op_module = load_inline(
    name="fused_op_module",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses min, bias addition, and scaling into a single custom CUDA kernel.
    The convolution operation is left as the standard PyTorch implementation, as it is highly optimized.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        # Use the standard, highly optimized Conv2d implementation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Store parameters for the fused operation
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Step 1: Perform convolution using the highly optimized PyTorch operator
        x = self.conv(x)
        
        # Step 2: Apply the fused operation using the custom CUDA kernel
        # The kernel handles: min(x, constant), add bias, multiply by scale
        x = fused_op_module.fused_op_cuda(x, self.bias, self.constant_value, self.scaling_factor)
        
        return x