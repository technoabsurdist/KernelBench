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
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int out_height,
    int out_width
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
    int b = tmp / out_channels;
    
    float sum = 0.0f;
    
    // Calculate input position that would contribute to this output position
    int h_start = (h_out + padding - (kernel_size - 1)) % stride;
    int w_start = (w_out + padding - (kernel_size - 1)) % stride;
    
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_in = (h_out + padding - kh) / stride;
            int w_in = (w_out + padding - kw) / stride;
            
            if ((h_out + padding - kh) % stride == 0 && 
                (w_out + padding - kw) % stride == 0 &&
                h_in >= 0 && h_in < in_height &&
                w_in >= 0 && w_in < in_width) {
                
                for (int c_in = 0; c_in < in_channels; c_in++) {
                    int input_idx = b * (in_channels * in_height * in_width) +
                                   c_in * (in_height * in_width) +
                                   h_in * in_width + w_in;
                                   
                    int weight_idx = c_out * (in_channels * kernel_size * kernel_size) +
                                    c_in * (kernel_size * kernel_size) +
                                    (kernel_size - 1 - kh) * kernel_size + 
                                    (kernel_size - 1 - kw);
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    output[out_idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * out_height * out_width;
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
        kernel_size,
        stride,
        padding,
        output_padding,
        out_height,
        out_width
    );
    
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding
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
    Performs a transposed 2D convolution with square input and square kernel using custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Only groups=1 is supported in this custom implementation")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_flag = bias
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        # Initialize bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.conv_transpose2d_func = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Call custom CUDA kernel
        output = self.conv_transpose2d_func.conv_transpose2d_cuda(
            x, self.weight, self.stride, self.padding, self.output_padding
        )
        
        # Add bias if needed
        if self.bias_flag:
            output += self.bias.view(1, -1, 1, 1)
            
        return output