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
    int height_in,
    int width_in,
    int height_out,
    int width_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height_out * width_out;
    
    if (out_idx >= total_elements) return;
    
    int w_out = out_idx % width_out;
    out_idx /= width_out;
    int h_out = out_idx % height_out;
    out_idx /= height_out;
    int c_out = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    float sum = 0.0f;
    
    // Calculate input position that would contribute to this output position
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            // Calculate corresponding input position
            int h_in = h_out + padding - kh;
            int w_in = w_out + padding - kw;
            
            // Check if within valid input bounds after accounting for stride
            if (h_in >= 0 && h_in < height_in * stride && h_in % stride == 0 &&
                w_in >= 0 && w_in < width_in * stride && w_in % stride == 0) {
                
                h_in /= stride;
                w_in /= stride;
                
                if (h_in < height_in && w_in < width_in) {
                    // Get input value
                    int input_idx = ((b * in_channels) + 0) * (height_in * width_in) + 
                                   h_in * width_in + w_in;
                    // Get weight value (simplified for single group)
                    int weight_idx = (0 * out_channels + c_out) * (kernel_size * kernel_size) + 
                                    kh * kernel_size + kw;
                    
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
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int height_in = input_sizes[2];
    int width_in = input_sizes[3];
    
    int out_channels = weight_sizes[1];
    int kernel_size = weight_sizes[2];
    
    // Calculate output dimensions
    int height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int threads_per_block = 256;
    const int num_elements = batch_size * out_channels * height_out * width_out;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_size,
        stride,
        padding,
        output_padding
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
    Performs a transposed 2D convolution with asymmetric input and a square kernel using custom CUDA kernel.

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
        self.conv_transpose2d_op = conv_transpose2d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        output = self.conv_transpose2d_op.conv_transpose2d_cuda(
            x, 
            self.weight, 
            self.stride, 
            self.padding, 
            self.output_padding
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
            
        return output