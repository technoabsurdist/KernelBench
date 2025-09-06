import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv2d_kernel(
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
    int dilation_h,
    int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_height * output_width;
    
    if (out_idx >= total_outputs) return;
    
    int w_out = out_idx % output_width;
    out_idx /= output_width;
    int h_out = out_idx % output_height;
    out_idx /= output_height;
    int c_out = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int input_idx = b * (in_channels * input_height * input_width) +
                                    c_in * (input_height * input_width) +
                                    h_in * input_width + w_in;
                    int weight_idx = c_out * (in_channels * kernel_height * kernel_width) +
                                     c_in * (kernel_height * kernel_width) +
                                     kh * kernel_width + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    output[out_idx * (out_channels * output_height * output_width) +
           c_out * (output_height * output_width) +
           h_out * output_width + w_out] = sum;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_height = input_sizes[2];
    int input_width = input_sizes[3];
    int out_channels = weight_sizes[0];
    int kernel_height = weight_sizes[2];
    int kernel_width = weight_sizes[3];
    
    int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
    
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, options);
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * output_height * output_width + block_size - 1) / block_size;
    
    conv2d_kernel<<<num_blocks, block_size>>>(
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
        dilation_h,
        dilation_w
    );
    
    // Add bias if present
    if (bias.defined()) {
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < out_channels; ++c) {
                float bias_val = bias[c].item<float>();
                for (int h = 0; h < output_height; ++h) {
                    for (int w = 0; w < output_width; ++w) {
                        output[b][c][h][w] += bias_val;
                    }
                }
            }
        }
    }
    
    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
);
"""

# Compile the inline CUDA code for 2D convolution
conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes using custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers representing the height and width of the convolution kernel.
        stride (tuple, optional): Tuple of two integers representing the stride in the height and width dimensions. Defaults to (1, 1).
        padding (tuple, optional): Tuple of two integers representing the padding in the height and width dimensions. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of two integers representing the dilation in the height and width dimensions. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Grouped convolutions are not supported in this custom implementation")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.conv2d = conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d.conv2d_cuda(
            x,
            self.weight,
            self.bias if self.use_bias else torch.tensor([]),
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1]
        )

# Test code
batch_size = 8
in_channels = 64
out_channels = 128
kernel_size = (5, 7)
height = 512
width = 256

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization