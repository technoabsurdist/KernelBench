import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv-scale-min operation
fused_conv_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernel for element-wise scaling and min reduction
__global__ void scale_and_min_kernel(const float* input, float* output, 
                                     float scale_factor, int batch_size, 
                                     int out_channels, int height, int width) {
    int batch_idx = blockIdx.x;
    int hw_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || hw_idx >= height * width) return;
    
    int spatial_size = height * width;
    float min_val = INFINITY;
    
    for (int c = 0; c < out_channels; ++c) {
        int idx = batch_idx * (out_channels * spatial_size) + 
                  c * spatial_size + hw_idx;
        float scaled_val = input[idx] * scale_factor;
        min_val = fminf(min_val, scaled_val);
    }
    
    output[batch_idx * spatial_size + hw_idx] = min_val;
}

torch::Tensor fused_conv_scale_min_cuda(torch::Tensor conv_output, float scale_factor) {
    auto batch_size = conv_output.size(0);
    auto out_channels = conv_output.size(1);
    auto height = conv_output.size(2);
    auto width = conv_output.size(3);
    
    // Output has shape (batch_size, 1, height, width)
    auto output = torch::zeros({batch_size, 1, height, width}, 
                               torch::TensorOptions().dtype(conv_output.dtype()).device(conv_output.device()));
    
    // Launch kernel
    dim3 grid(batch_size, (height * width + 255) / 256);
    dim3 block(256);
    
    scale_and_min_kernel<<<grid, block>>>(
        conv_output.data_ptr<float>(), 
        output.data_ptr<float>(),
        scale_factor,
        batch_size,
        out_channels,
        height,
        width
    );
    
    return output;
}
"""

fused_conv_scale_min_cpp_source = """
torch::Tensor fused_conv_scale_min_cuda(torch::Tensor conv_output, float scale_factor);
"""

# Compile the inline CUDA code for fused operation
fused_conv_scale_min = load_inline(
    name="fused_conv_scale_min",
    cpp_sources=fused_conv_scale_min_cpp_source,
    cuda_sources=fused_conv_scale_min_source,
    functions=["fused_conv_scale_min_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused conv-scale-min operation using custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.fused_op = fused_conv_scale_min

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, height, width).
        """
        x = self.conv(x)
        return self.fused_op.fused_conv_scale_min_cuda(x, self.scale_factor)