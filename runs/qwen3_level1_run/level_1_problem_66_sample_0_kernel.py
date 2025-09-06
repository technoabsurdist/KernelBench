import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_outputs) return;
    
    int w_out = out_idx % output_w;
    out_idx /= output_w;
    int h_out = out_idx % output_h;
    out_idx /= output_h;
    int d_out = out_idx % output_d;
    out_idx /= output_d;
    int c_out = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int d_in = d_out * stride_d - padding_d + kd * dilation_d;
                    int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                    
                    if (d_in >= 0 && d_in < input_d &&
                        h_in >= 0 && h_in < input_h &&
                        w_in >= 0 && w_in < input_w) {
                        
                        int input_idx = b * (in_channels * input_d * input_h * input_w) +
                                       c_in * (input_d * input_h * input_w) +
                                       d_in * (input_h * input_w) +
                                       h_in * input_w +
                                       w_in;
                                       
                        int weight_idx = c_out * (in_channels * kernel_d * kernel_h * kernel_w) +
                                        c_in * (kernel_d * kernel_h * kernel_w) +
                                        kd * (kernel_h * kernel_w) +
                                        kh * kernel_w +
                                        kw;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[out_idx * (out_channels * output_d * output_h * output_w) +
           c_out * (output_d * output_h * output_w) +
           d_out * (output_h * output_w) +
           h_out * output_w +
           w_out] = sum;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation
) {
    // Extract dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_d = input.size(2);
    auto input_h = input.size(3);
    auto input_w = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    // Calculate output dimensions
    auto output_d = (input_d + 2 * padding[0] - dilation[0] * (kernel_d - 1) - 1) / stride[0] + 1;
    auto output_h = (input_h + 2 * padding[1] - dilation[1] * (kernel_h - 1) - 1) / stride[1] + 1;
    auto output_w = (input_w + 2 * padding[2] - dilation[2] * (kernel_w - 1) - 1) / stride[2] + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (batch_size * out_channels * output_d * output_h * output_w + threads_per_block - 1) / threads_per_block;
    
    conv3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2]
    );
    
    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation
);
"""

# Compile the inline CUDA code for 3D convolution
conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.
    Optimized with custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Use PyTorch's Conv3d to initialize weights
        self.conv3d_ref = nn.Conv3d(in_channels, out_channels, kernel_size, 
                                   stride=stride, padding=padding, 
                                   dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Use custom CUDA implementation for forward pass
        output = conv3d.conv3d_cuda(x, self.conv3d_ref.weight, self.stride, self.padding, self.dilation)
        
        # Add bias if needed
        if self.bias:
            output += self.conv3d_ref.bias.view(1, -1, 1, 1, 1)
            
        return output