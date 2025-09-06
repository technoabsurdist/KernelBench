import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise convolution with asymmetric kernel
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int out_height,
    int out_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (out_x >= out_width || out_y >= out_height || ch >= in_channels) return;
    
    int batch_idx = 0;  // Handle one batch at a time in the kernel
    float sum = 0.0f;
    
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int in_y = out_y * stride_h - padding_h + kh * dilation_h;
            int in_x = out_x * stride_w - padding_w + kw * dilation_w;
            
            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                int input_idx = ((batch_idx * in_channels + ch) * height + in_y) * width + in_x;
                int weight_idx = (ch * kernel_h + kh) * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    int output_idx = ((batch_idx * in_channels + ch) * out_height + out_y) * out_width + out_x;
    output[output_idx] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
) {
    // Ensure we're on the correct device
    at::cuda::CUDAGuard device_guard(input.device());
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int height = input_sizes[2];
    int width = input_sizes[3];
    
    int kernel_h = weight_sizes[2];
    int kernel_w = weight_sizes[3];
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Launch kernel for each batch item
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        auto input_slice = input[batch_idx];
        auto output_slice = output[batch_idx];
        
        const int block_size_x = 16;
        const int block_size_y = 16;
        const int block_size_z = 4;
        
        dim3 block_size(block_size_x, block_size_y, block_size_z);
        dim3 grid_size(
            (out_width + block_size_x - 1) / block_size_x,
            (out_height + block_size_y - 1) / block_size_y,
            (in_channels + block_size_z - 1) / block_size_z
        );
        
        depthwise_conv2d_kernel<<<grid_size, block_size>>>(
            input_slice.data_ptr<float>(),
            weight.data_ptr<float>(),
            output_slice.data_ptr<float>(),
            1,  // Single batch item
            in_channels,
            height,
            width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            out_height,
            out_width
        );
    }
    
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
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
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel using custom CUDA implementation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Create weight parameter for depthwise convolution with asymmetric kernel (kernel_size x 1)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        
        if bias:
            self.bias_param = nn.Parameter(torch.randn(in_channels))
        else:
            self.bias_param = None
            
        self.depthwise_conv2d = depthwise_conv2d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Call our custom CUDA kernel
        output = self.depthwise_conv2d.depthwise_conv2d_cuda(
            x,
            self.weight,
            self.stride,
            1,  # stride_w = 1 for (kernel_size, 1) kernel
            self.padding,
            0,  # padding_w = 0 for (kernel_size, 1) kernel
            self.dilation,
            1   # dilation_w = 1 for (kernel_size, 1) kernel
        )
        
        # Add bias if needed
        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1, 1)
            
        return output