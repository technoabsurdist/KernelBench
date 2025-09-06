import torch
import torch.nn as nn
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
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_elements) return;
    
    int tmp = out_idx;
    int w_out = tmp % out_width;
    tmp /= out_width;
    int h_out = tmp % out_height;
    tmp /= out_height;
    int c_out = tmp % out_channels;
    int n = tmp / out_channels;
    
    float sum = 0.0f;
    
    // Calculate input position that would contribute to this output position
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            // Calculate corresponding input position
            int h_in = (h_out + padding_h - kh) / stride_h;
            int w_in = (w_out + padding_w - kw) / stride_w;
            
            // Check if the division was exact (i.e., this input position contributes)
            if ((h_out + padding_h - kh) % stride_h == 0 && 
                (w_out + padding_w - kw) % stride_w == 0 &&
                h_in >= 0 && h_in < in_height &&
                w_in >= 0 && w_in < in_width) {
                
                // Get input value
                int input_idx = ((n * in_channels) + c_in) * (in_height * in_width) + h_in * in_width + w_in;
                
                // For simplicity, assuming groups=1 (in_channels == out_channels)
                int c_in = c_out; // Simplified for groups=1
                
                // Get weight value
                int weight_idx = (c_in * out_channels + c_out) * (kernel_h * kernel_w) + kh * kernel_w + kw;
                
                sum += input[input_idx] * weight[weight_idx];
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
    // Set the CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = weight.size(1);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    auto stride_h = stride[0];
    auto stride_w = stride[1];
    auto padding_h = padding[0];
    auto padding_w = padding[1];
    auto output_padding_h = output_padding[0];
    auto output_padding_w = output_padding[1];
    
    auto out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    auto out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    auto total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w
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
    Performs a transposed 2D convolution with a square input and an asymmetric kernel using custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
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
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        
        # Initialize bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        output = conv_transpose2d.conv_transpose2d_cuda(x, self.weight, self.stride, self.padding, self.output_padding)
        
        # Add bias if it exists
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
            
        return output