import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scale, maxpool, and clamp
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_scale_maxpool_clamp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    float* __restrict__ out,
    const int N,
    const int C,
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const int K, // maxpool_kernel_size
    const float clamp_min,
    const float clamp_max) {

    const int total_output_elements = N * C * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_output_elements) {
        // Map 1D index to 4D output coordinates (n, c, h_out, w_out)
        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int c = (idx / (W_out * H_out)) % C;
        const int n = idx / (W_out * H_out * C);

        // Determine the top-left corner of the pooling window in the input tensor
        const int h_start = h_out * K;
        const int w_start = w_out * K;

        float current_max = -FLT_MAX;
        const float scale_val = scale[c]; // scale is (C, 1, 1), so we can index by channel

        // Iterate over the pooling window
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int h_in = h_start + kh;
                const int w_in = w_start + kw;

                // Calculate the 1D index for the input tensor
                const int in_idx = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                
                // Apply scale and find the max value
                current_max = fmaxf(current_max, x[in_idx] * scale_val);
            }
        }

        // Apply clamp and write to output
        out[idx] = fmaxf(clamp_min, fminf(current_max, clamp_max));
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    int K,
    float clamp_min,
    float clamp_max) {
    
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "Scale tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "Scale tensor must be contiguous");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);

    TORCH_CHECK(H_in % K == 0 && W_in % K == 0, "Input dimensions must be divisible by maxpool kernel size");

    const int H_out = H_in / K;
    const int W_out = W_in / K;

    auto out = torch::empty({N, C, H_out, W_out}, x.options());
    const int total_output_elements = out.numel();

    if (total_output_elements == 0) {
        return out;
    }

    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;

    fused_scale_maxpool_clamp_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out, K,
        clamp_min, clamp_max
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor scale, int K, float clamp_min, float clamp_max);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs convolution and group normalization using standard PyTorch operators,
    and then applies a fused custom CUDA kernel for scaling, max pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Use standard PyTorch for complex operations like Conv and GroupNorm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        
        # Keep the scale parameter
        self.scale = nn.Parameter(torch.ones(scale_shape))
        
        # Store parameters for the custom kernel
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        x = self.group_norm(x)
        
        # Call the custom fused kernel for scale, maxpool, and clamp
        x = fused_op.fused_op_cuda(
            x, 
            self.scale, 
            self.maxpool_kernel_size, 
            self.clamp_min, 
            self.clamp_max
        )
        return x