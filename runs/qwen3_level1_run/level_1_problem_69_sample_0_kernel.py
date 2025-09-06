import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 2D convolution
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
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int dilation_h,
    int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int tmp = out_idx;
    int w_out = tmp % output_width;
    tmp /= output_width;
    int h_out = tmp % output_height;
    tmp /= output_height;
    int c_out = tmp % out_channels;
    int n = tmp / out_channels;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int h_in = h_out + padding_h - kh * dilation_h;
                int w_in = w_out + padding_w - kw * dilation_w;
                
                if (h_in % stride_h == 0 && w_in % stride_w == 0) {
                    h_in /= stride_h;
                    w_in /= stride_w;
                    
                    if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                        int input_idx = ((n * in_channels + c_in) * input_height + h_in) * input_width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_height + kh) * kernel_width + kw;
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
    torch::IntArrayRef output_padding,
    torch::IntArrayRef dilation
) {
    // Set the CUDA device guard to ensure operations happen on the correct device
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_height = input_sizes[2];
    int input_width = input_sizes[3];
    
    int out_channels = weight_sizes[0];
    int kernel_height = weight_sizes[2];
    int kernel_width = weight_sizes[3];
    
    int stride_h = stride[0];
    int stride_w = stride[1];
    int padding_h = padding[0];
    int padding_w = padding[1];
    int output_padding_h = output_padding[0];
    int output_padding_w = output_padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];
    
    int output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + output_padding_h + 1;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + output_padding_w + 1;
    
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
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        dilation_h,
        dilation_w
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
    torch::IntArrayRef output_padding,
    torch::IntArrayRef dilation
);
"""

# Compile the inline CUDA code for transposed 2D convolution
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
    Performs a transposed 2D convolution operation with asymmetric input and kernel size using custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
        output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Use PyTorch's ConvTranspose2d to create the weight and bias parameters
        self.conv_transpose2d_ref = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, dilation=dilation, groups=groups, bias=bias
        )
        
        # Register the custom CUDA function
        self.conv_transpose2d_cuda = conv_transpose2d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use the custom CUDA kernel for the convolution operation
        output = self.conv_transpose2d_cuda.conv_transpose2d_cuda(
            x, 
            self.conv_transpose2d_ref.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation
        )
        
        # Add bias if needed
        if self.bias:
            output += self.conv_transpose2d_ref.bias.view(1, -1, 1, 1)
        
        return output