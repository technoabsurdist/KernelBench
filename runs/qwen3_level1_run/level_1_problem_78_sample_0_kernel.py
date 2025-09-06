import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    int pad_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int n = out_idx / (out_channels * output_height * output_width);
    int c_out = (out_idx / (output_height * output_width)) % out_channels;
    int h_out = (out_idx / output_width) % output_height;
    int w_out = out_idx % output_width;
    
    float sum = 0.0f;
    
    // Iterate through kernel
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                // Calculate corresponding input position
                int h_in = h_out + pad_h - kh;
                int w_in = w_out + pad_w - kw;
                
                // Check bounds and stride
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    // Apply stride condition
                    if (h_in % stride_h == 0 && w_in % stride_w == 0) {
                        h_in /= stride_h;
                        w_in /= stride_w;
                        
                        int input_idx = n * (in_channels * input_height * input_width) +
                                       c_in * (input_height * input_width) +
                                       h_in * input_width + w_in;
                                       
                        int weight_idx = c_in * (out_channels * kernel_height * kernel_width) +
                                        c_out * (kernel_height * kernel_width) +
                                        kh * kernel_width + kw;
                                        
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
    torch::IntArrayRef padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    
    int out_channels = weight.size(1);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);
    
    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];
    
    int output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height;
    int output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
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
        pad_w
    );
    
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding
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
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.
    Optimized with custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None
            
        self.conv_transpose2d_op = conv_transpose2d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        output = self.conv_transpose2d_op.conv_transpose2d_cuda(x, self.weight, self.stride, self.padding)
        
        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1, 1)
            
        return output