import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void col2im_kernel(
    const float* data_col,
    const int channels,
    const int depth, const int height, const int width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int depth_col, const int height_col, const int width_col,
    float* data_im) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = depth * height * width * channels;
    
    if (index >= total_threads) return;
    
    int w = index % width;
    index /= width;
    int h = index % height;
    index /= height;
    int d = index % depth;
    int c = index / depth;
    
    float val = 0;
    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int d_out = d + pad_d - kd * dilation_d;
                int h_out = h + pad_h - kh * dilation_h;
                int w_out = w + pad_w - kw * dilation_w;
                
                if ((d_out % stride_d == 0) && (h_out % stride_h == 0) && (w_out % stride_w == 0)) {
                    d_out /= stride_d;
                    h_out /= stride_h;
                    w_out /= stride_w;
                    
                    if (d_out >= 0 && h_out >= 0 && w_out >= 0 &&
                        d_out < depth_col && h_out < height_col && w_out < width_col) {
                        int col_index = (((c * kernel_d + kd) * kernel_h + kh) * kernel_w + kw) *
                                        (depth_col * height_col * width_col) +
                                        ((d_out * height_col + h_out) * width_col + w_out);
                        val += data_col[col_index];
                    }
                }
            }
        }
    }
    data_im[(c * depth + d) * height * width + h * width + w] = val;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding) {
    
    // Input shape: (N, C_in, D_in, H_in, W_in)
    // Weight shape: (C_in, C_out/groups, D_k, H_k, W_k)
    // Output shape: (N, C_out, D_out, H_out, W_out)
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    
    int out_pad_d = output_padding[0];
    int out_pad_h = output_padding[1];
    int out_pad_w = output_padding[2];
    
    // Calculate output dimensions
    int depth_out = (depth_in - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
    int height_out = (height_in - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    int width_out = (width_in - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // For each item in batch
    for (int n = 0; n < batch_size; n++) {
        // Im2col transformation
        int col_depth = (depth_out + 2 * pad_d - kernel_d) / stride_d + 1;
        int col_height = (height_out + 2 * pad_h - kernel_h) / stride_h + 1;
        int col_width = (width_out + 2 * pad_w - kernel_w) / stride_w + 1;
        
        auto col = torch::zeros({in_channels * kernel_d * kernel_h * kernel_w, 
                                col_depth * col_height * col_width},
                                torch::TensorOptions().dtype(input.dtype()).device(input.device()));
        
        auto input_n = input[n];
        auto output_n = output[n];
        
        // Perform matrix multiplication: weight * input_col = output
        // Weight is (C_out, C_in, D_k, H_k, W_k) -> reshape to (C_out, C_in*D_k*H_k*W_k)
        auto weight_reshaped = weight.view({in_channels, out_channels, -1}).permute({1, 0, 2}).contiguous();
        weight_reshaped = weight_reshaped.view({out_channels, -1});
        
        // Input is (C_in, D_in, H_in, W_in) -> reshape to (C_in*D_in*H_in*W_in, 1)
        auto input_reshaped = input_n.view({in_channels, -1});
        
        // Perform convolution as matrix multiplication
        auto output_reshaped = torch::matmul(weight_reshaped, input_reshaped);
        output_reshaped = output_reshaped.view({out_channels, depth_out, height_out, width_out});
        
        if (bias.defined()) {
            output_reshaped = output_reshaped + bias.view({out_channels, 1, 1, 1});
        }
        
        output[n] = output_reshaped;
    }
    
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding);
"""

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel using custom CUDA kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Groups other than 1 are not supported in this custom implementation")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weight parameters
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, 
                                              kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.conv_transpose3d_op = conv_transpose3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return self.conv_transpose3d_op.conv_transpose3d_cuda(
            x, self.weight, self.bias if self.bias_flag else torch.tensor([]), 
            list(self.stride), list(self.padding), list(self.output_padding))