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
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % out_width;
    out_idx /= out_width;
    int h = out_idx % out_height;
    out_idx /= out_height;
    int c = out_idx % out_channels;
    int n = out_idx / out_channels;
    
    float sum = 0.0f;
    
    int group_id = c * groups / out_channels;
    int in_ch_per_group = in_channels / groups;
    int out_ch_per_group = out_channels / groups;
    
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int in_h = h - kh * dilation_h + 2 * pad_h;
            int in_w = w - kw * dilation_w + 2 * pad_w;
            
            if (in_h % stride_h == 0 && in_w % stride_w == 0) {
                in_h /= stride_h;
                in_w /= stride_w;
                
                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    for (int g = 0; g < groups; ++g) {
                        if (c >= g * out_ch_per_group && c < (g + 1) * out_ch_per_group) {
                            for (int ic = g * in_ch_per_group; ic < (g + 1) * in_ch_per_group; ++ic) {
                                int weight_idx = ((c - g * out_ch_per_group) * in_ch_per_group + (ic - g * in_ch_per_group)) * kernel_h * kernel_w + 
                                                 (kernel_h - 1 - kh) * kernel_w + (kernel_w - 1 - kw);
                                int input_idx = ((n * in_channels + ic) * in_height + in_h) * in_width + in_w;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    
    output[out_idx * out_height * out_width + h * out_width + w] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    int64_t groups
) {
    // Set the CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_height = input_sizes[2];
    int in_width = input_sizes[3];
    
    int out_channels = weight_sizes[1];
    int kernel_h = weight_sizes[2];
    int kernel_w = weight_sizes[3];
    
    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];
    
    int out_height = (in_height - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    int out_width = (in_width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * out_channels * out_height * out_width;
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
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups
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
    torch::IntArrayRef dilation,
    int64_t groups
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
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel, 
    grouped, padded, and dilated using custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        
        # Initialize bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
        self.conv_transpose2d_op = conv_transpose2d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        output = self.conv_transpose2d_op.conv_transpose2d_cuda(
            x, 
            self.weight, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )
        
        if self.bias is not None:
            # Add bias if present
            output += self.bias.view(1, -1, 1, 1)
            
        return output