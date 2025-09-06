import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fusing tanh, scaling, and bias addition
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // for tanhf

__global__ void fused_tanh_scale_bias_kernel(
    const float* in,
    float* out,
    const float scaling_factor,
    const float* bias,
    const int size,
    const int channels,
    const int height,
    const int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Calculate the channel index from the linear index for broadcasting the bias
        // The memory layout is NCHW.
        // idx = n * C*H*W + c * H*W + h * W + w
        // idx / (H*W) = n*C + c
        // (idx / (H*W)) % C = c
        int c = (idx / (height * width)) % channels;

        // Fused operation: tanh -> scale -> bias_add
        out[idx] = tanhf(in[idx]) * scaling_factor + bias[c];
    }
}

torch::Tensor fused_tanh_scale_bias_cuda(
    torch::Tensor x,
    const float scaling_factor,
    torch::Tensor bias)
{
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input tensor 'bias' must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor 'x' must be contiguous");
    // Bias is small, contiguity check is less critical but good practice
    TORCH_CHECK(bias.is_contiguous(), "Input tensor 'bias' must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input tensor 'x' must be 4-dimensional (NCHW)");
    
    const auto channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);
    const auto size = x.numel();

    // Create an output tensor of the same shape as the input
    auto out = torch::empty_like(x);

    // Configure and launch the kernel
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_tanh_scale_bias_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        scaling_factor,
        bias.data_ptr<float>(),
        size,
        channels,
        height,
        width
    );
    
    // Check for any CUDA errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_tanh_scale_bias_cuda(
    torch::Tensor x,
    const float scaling_factor,
    torch::Tensor bias);
"""

# JIT compile the custom CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_tanh_scale_bias_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    A model that performs a convolution, applies a fused (tanh -> scaling -> bias) operation,
    and then max-pools. The fused operation is implemented as a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # 1. Convolution (unchanged, uses PyTorch's optimized implementation)
        x = self.conv(x)
        
        # 2. Fused operation: tanh -> scaling -> bias addition
        # This replaces three separate operations with a single kernel launch.
        x = fused_op.fused_tanh_scale_bias_cuda(x, self.scaling_factor, self.bias)
        
        # 3. Max-pooling (unchanged)
        x = self.max_pool(x)
        
        return x