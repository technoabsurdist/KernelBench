import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
# This kernel fuses the following sequence of operations into a single pass:
# original_x = x
# x = x + self.bias
# x = x + original_x
# x = x * original_x
# x = x + original_x
# The equivalent element-wise computation is: out = (2*x + bias)*x + x
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel(
    const float* x, 
    const float* bias, 
    float* out, 
    long long int size, 
    int out_channels, 
    int depth, 
    int height, 
    int width) {
    
    long long int idx = (long long int)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Calculate the size of a single feature map (D*H*W)
        long long int feature_map_size = (long long int)depth * height * width;
        
        // Calculate the channel index for the current element to handle bias broadcasting
        int channel_idx = (idx / feature_map_size) % out_channels;
        
        float x_val = x[idx];
        float bias_val = bias[channel_idx];
        
        // Fused operation:
        // original_x = x
        // temp = x + bias
        // temp = temp + original_x  => (x + bias) + x => 2*x + bias
        // temp = temp * original_x  => (2*x + bias) * x => 2*x^2 + bias*x
        // out = temp + original_x   => (2*x^2 + bias*x) + x
        out[idx] = (2.0f * x_val + bias_val) * x_val + x_val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input tensor bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Input tensor bias must be contiguous");

    auto out = torch::empty_like(x);
    auto size = x.numel();

    // Get dimensions for channel index calculation
    // x shape: (N, C, D, H, W)
    const int out_channels = x.size(1);
    const int depth = x.size(2);
    const int height = x.size(3);
    const int width = x.size(4);

    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_op_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size,
        out_channels,
        depth,
        height,
        width
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for the fused operation
# This might take a moment the first time it's run.
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a custom fused operation.
    The fused operation combines: bias add, residual add, multiplication, and another residual add.
    This approach reduces kernel launch overhead and memory bandwidth usage by avoiding intermediate tensors.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        # Use PyTorch's highly optimized cuDNN backend for the convolution
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Store the compiled custom operator
        self.fused_op = fused_op

    def forward(self, x):
        # Step 1: Perform the expensive convolution using PyTorch's implementation
        x = self.conv_transpose(x)
        
        # Step 2: Perform the sequence of element-wise operations in a single fused CUDA kernel
        x = self.fused_op.fused_op_cuda(x, self.bias)
        
        return x