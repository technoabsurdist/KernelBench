import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for pointwise convolution (1x1 conv2d)
pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void pointwise_conv_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    bool has_bias
) {
    int batch_idx = blockIdx.x;
    int out_ch_idx = blockIdx.y;
    int pixel_idx = threadIdx.x + blockIdx.z * blockDim.x;
    
    if (batch_idx >= batch_size || out_ch_idx >= out_channels) return;
    
    int hw = height * width;
    if (pixel_idx >= hw) return;
    
    float sum = 0.0f;
    
    // Perform 1x1 convolution (equivalent to matrix multiplication over channels)
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        sum += input[((batch_idx * in_channels + in_ch) * height + pixel_idx / width) * width + pixel_idx % width] *
               weight[out_ch_idx * in_channels + in_ch];
    }
    
    if (has_bias) {
        sum += bias[out_ch_idx];
    }
    
    output[((batch_idx * out_channels + out_ch_idx) * height + pixel_idx / width) * width + pixel_idx % width] = sum;
}

torch::Tensor pointwise_conv_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool has_bias
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int height = input_sizes[2];
    int width = input_sizes[3];
    int out_channels = weight_sizes[0];
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int threads_per_block = 256;
    const int pixels_per_batch = height * width;
    const int blocks_per_grid_z = (pixels_per_batch + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, out_channels, blocks_per_grid_z);
    dim3 block(threads_per_block);
    
    pointwise_conv_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        has_bias
    );
    
    return output;
}
"""

pointwise_conv_cpp_source = """
torch::Tensor pointwise_conv_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool has_bias
);
"""

# Compile the inline CUDA code for pointwise convolution
pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=pointwise_conv_cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized version of pointwise 2D convolution using custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = bias
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        
        # Initialize bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Load custom CUDA function
        self.pointwise_conv = pointwise_conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.pointwise_conv.pointwise_conv_cuda(
            x, 
            self.weight, 
            self.bias if self.has_bias else torch.empty(0, device=x.device), 
            self.has_bias
        )