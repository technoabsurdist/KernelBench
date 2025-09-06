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
    int total_output_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_output_elements) return;
    
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
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation
) {
    // Extract dimensions
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];
    
    int out_channels = weight_sizes[0];
    int kernel_d = weight_sizes[2];
    int kernel_h = weight_sizes[3];
    int kernel_w = weight_sizes[4];
    
    // Stride, padding, dilation
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    
    int padding_d = padding[0];
    int padding_h = padding[1];
    int padding_w = padding[2];
    
    int dilation_d = dilation[0];
    int dilation_h = dilation[1];
    int dilation_w = dilation[2];
    
    // Calculate output dimensions
    int output_d = (input_d + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int output_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Launch kernel
    int total_output_elements = batch_size * out_channels * output_d * output_h * output_w;
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;
    
    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w
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
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel,
    optimized with custom CUDA kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.bias = bias
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        
        # Initialize bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
        self.conv3d_op = conv3d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        # Use custom CUDA kernel for convolution
        output = self.conv3d_op.conv3d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        
        # Add bias if needed
        if self.bias is not None:
            # Reshape bias to broadcast correctly
            bias_view = self.bias.view(1, -1, 1, 1, 1)
            output = output + bias_view
            
        return output