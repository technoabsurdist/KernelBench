import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LogSumExp + HardSwish
fused_lse_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>

__global__ void fused_lse_hardswish_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int batch_idx = blockIdx.x;
    int spatial_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || spatial_idx >= spatial_size) return;
    
    // Compute LogSumExp for this spatial location across channels
    const float* input_ptr = input + batch_idx * channels * spatial_size + spatial_idx;
    
    // Find max for numerical stability
    float max_val = input_ptr[0];
    for (int c = 1; c < channels; c++) {
        float val = input_ptr[c * spatial_size];
        max_val = fmaxf(max_val, val);
    }
    
    // Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int c = 0; c < channels; c++) {
        sum_exp += expf(input_ptr[c * spatial_size] - max_val);
    }
    
    // Compute logsumexp
    float lse = max_val + logf(sum_exp);
    
    // Apply HardSwish: x * sigmoid(x + 3) / 6
    float sigmoid_val = 1.0f / (1.0f + expf(-(lse + 3.0f)));
    float hardswish_val = lse * sigmoid_val / 6.0f;
    
    // Write output
    output[batch_idx * spatial_size + spatial_idx] = hardswish_val;
}
"""

# Define the custom CUDA kernel for fused subtraction and clamp
sub_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void sub_clamp_kernel(
    const float* input,
    const float bias,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] - bias;
        output[idx] = fmaxf(-1.0f, fminf(1.0f, val));
    }
}
"""

fused_lse_hardswish_cpp_source = """
torch::Tensor fused_lse_hardswish_cuda(torch::Tensor input);
"""

sub_clamp_cpp_source = """
torch::Tensor sub_clamp_cuda(torch::Tensor input, float bias);
"""

# Compile the inline CUDA code
fused_lse_hardswish = load_inline(
    name="fused_lse_hardswish",
    cpp_sources=fused_lse_hardswish_cpp_source,
    cuda_sources=fused_lse_hardswish_source,
    functions=["fused_lse_hardswish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

sub_clamp = load_inline(
    name="sub_clamp",
    cpp_sources=sub_clamp_cpp_source,
    cuda_sources=sub_clamp_source,
    functions=["sub_clamp_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_lse_hardswish = fused_lse_hardswish
        self.sub_clamp = sub_clamp

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_lse_hardswish.fused_lse_hardswish_cuda(x)
        x = x.unsqueeze(1)  # Add channel dimension back
        x = self.sub_clamp.sub_clamp_cuda(x, self.bias.item())
        return x