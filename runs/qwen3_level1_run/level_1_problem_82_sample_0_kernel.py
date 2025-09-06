import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int out_height,
    int out_width,
    int stride,
    int padding,
    bool has_bias
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (out_x >= out_width || out_y >= out_height || ch >= in_channels) return;
    
    int batch_idx = blockIdx.z / in_channels;
    int channel_idx = blockIdx.z % in_channels;
    
    float sum = 0.0f;
    
    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            int in_y = out_y * stride - padding + ky;
            int in_x = out_x * stride - padding + kx;
            
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int input_idx = batch_idx * (in_channels * in_height * in_width) + 
                               channel_idx * (in_height * in_width) + 
                               in_y * in_width + in_x;
                int weight_idx = channel_idx * (kernel_size * kernel_size) + ky * kernel_size + kx;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    if (has_bias) {
        sum += bias[channel_idx];
    }
    
    int output_idx = batch_idx * (in_channels * out_height * out_width) + 
                    channel_idx * (out_height * out_width) + 
                    out_y * out_width + out_x;
    output[output_idx] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    bool has_bias
) {
    // Input shape: (batch_size, in_channels, in_height, in_width)
    // Weight shape: (in_channels, 1, kernel_size, kernel_size)
    // Bias shape: (in_channels)
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto kernel_size = weight.size(2);
    
    auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int block_size_x = 16;
    const int block_size_y = 16;
    const int block_size_z = 4;
    
    dim3 block_size(block_size_x, block_size_y, block_size_z);
    dim3 grid_size(
        (out_width + block_size_x - 1) / block_size_x,
        (out_height + block_size_y - 1) / block_size_y,
        batch_size * in_channels
    );
    
    // Ensure we're on the right device
    at::cuda::CUDAGuard device_guard(input.device());
    
    depthwise_conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        kernel_size,
        out_height,
        out_width,
        stride,
        padding,
        has_bias
    );
    
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    bool has_bias
);
"""

# Compile the inline CUDA code for depthwise convolution
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel using custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        
        # Depthwise convolution weights: (in_channels, 1, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
        self.depthwise_conv2d = depthwise_conv2d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return self.depthwise_conv2d.depthwise_conv2d_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.empty(0, device=x.device), 
            self.stride, 
            self.padding, 
            self.bias_flag
        )