import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias_add, scale, and sigmoid
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel to perform fused (x + bias) * scale -> sigmoid
// Bias and scale are broadcasted along the channel dimension.
__global__ void fused_bias_scale_sigmoid_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ out,
    const int total_elements,
    const int C,
    const int plane_size) {

    // Using a grid-stride loop for flexibility and efficiency
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        // Calculate the channel index for broadcasting bias and scale
        const int c = (idx / plane_size) % C;

        // Apply the fused operations
        float val = x[idx];
        val = (val + bias[c]) * scale[c];
        out[idx] = 1.0f / (1.0f + expf(-val));
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    torch::Tensor scale) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input tensor 'bias' must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "Input tensor 'scale' must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input tensor 'x' must be 4D (NCHW)");
    TORCH_CHECK(bias.dim() == 3 && bias.size(1) == 1 && bias.size(2) == 1, "Bias tensor must be (C, 1, 1)");
    TORCH_CHECK(scale.dim() == 3 && scale.size(1) == 1 && scale.size(2) == 1, "Scale tensor must be (C, 1, 1)");
    TORCH_CHECK(x.size(1) == bias.size(0), "Channel dimension of x and bias must match");
    TORCH_CHECK(x.size(1) == scale.size(0), "Channel dimension of x and scale must match");

    // Ensure tensors are contiguous
    x = x.contiguous();
    // Squeezing bias and scale to 1D simplifies indexing in the kernel.
    auto bias_1d = bias.contiguous().view({-1});
    auto scale_1d = scale.contiguous().view({-1});

    // Get tensor dimensions
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int total_elements = x.numel();
    const int plane_size = H * W;

    // Create output tensor
    auto out = torch::empty_like(x);

    // Configure and launch the kernel
    const int block_size = 256;
    // Heuristic for grid size. Can be tuned.
    const int num_blocks = std::min((total_elements + block_size - 1) / block_size, 4096);

    fused_bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias_1d.data_ptr<float>(),
        scale_1d.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements,
        C,
        plane_size);

    // Check for any CUDA errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature
fused_op_cpp_source = """
torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    torch::Tensor scale);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_bias_scale_sigmoid_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    The bias_add, scale, and sigmoid operations are fused into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

        # Store the custom fused operator
        self.fused_bias_scale_sigmoid = fused_op.fused_bias_scale_sigmoid_cuda

    def forward(self, x):
        # 1. Standard PyTorch convolution
        x = self.conv(x)

        # 2. Fused operation: x = sigmoid((x + bias) * scale)
        x = self.fused_bias_scale_sigmoid(x, self.bias, self.scale)

        # 3. Standard PyTorch GroupNorm
        x = self.group_norm(x)
        return x