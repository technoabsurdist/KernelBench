import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int output_pad_h,
    int output_pad_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int n = out_idx / (out_channels * output_height * output_width);
    int c_out = (out_idx / (output_height * output_width)) % out_channels;
    int out_y = (out_idx / output_width) % output_height;
    int out_x = out_idx % output_width;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int k_y = 0; k_y < kernel_height; k_y++) {
            for (int k_x = 0; k_x < kernel_width; k_x++) {
                int in_y = out_y - k_y + pad_h;
                int in_x = out_x - k_x + pad_w;
                
                if (in_y % stride_h == 0 && in_x % stride_w == 0) {
                    in_y /= stride_h;
                    in_x /= stride_w;
                    
                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                        int input_idx = n * (in_channels * input_height * input_width) +
                                       c_in * (input_height * input_width) +
                                       in_y * input_width + in_x;
                                       
                        int weight_idx = c_in * (out_channels * kernel_height * kernel_width) +
                                        c_out * (kernel_height * kernel_width) +
                                        (kernel_height - 1 - k_y) * kernel_width +
                                        (kernel_width - 1 - k_x);
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[out_idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    
    const auto out_channels = weight.size(1);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    const auto stride_h = stride[0];
    const auto stride_w = stride[1];
    const auto pad_h = padding[0];
    const auto pad_w = padding[1];
    const auto output_pad_h = output_padding[0];
    const auto output_pad_w = output_padding[1];
    
    const auto output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height + output_pad_h;
    const auto output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width + output_pad_w;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    at::cuda::CUDAGuard device_guard(input.device());
    
    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        output_pad_h,
        output_pad_w
    );
    
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding
);
"""

# Compile the inline CUDA code for transposed convolution
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.
    Optimized with custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1]))
        
        # Create bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
        # Load custom CUDA function
        self.conv_transpose2d_cuda = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        output = self.conv_transpose2d_cuda.conv_transpose2d_cuda(
            x, 
            self.weight, 
            self.stride, 
            self.padding, 
            self.output_padding
        )
        
        if self.bias is not None:
            # Add bias if present
            output += self.bias.view(1, -1, 1, 1)
            
        return output