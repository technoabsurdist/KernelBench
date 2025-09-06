import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int tmp = out_idx;
    int w_out = tmp % output_width;
    tmp /= output_width;
    int h_out = tmp % output_height;
    tmp /= output_height;
    int d_out = tmp % output_depth;
    tmp /= output_depth;
    int c_out = tmp % out_channels;
    int n = tmp / out_channels;
    
    float sum = 0.0f;
    
    for (int kd = 0; kd < kernel_depth; kd++) {
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                int d_in = d_out + padding_d - kd;
                int h_in = h_out + padding_h - kh;
                int w_in = w_out + padding_w - kw;
                
                if (d_in % stride_d == 0 && h_in % stride_h == 0 && w_in % stride_w == 0) {
                    d_in /= stride_d;
                    h_in /= stride_h;
                    w_in /= stride_w;
                    
                    if (d_in >= 0 && d_in < input_depth &&
                        h_in >= 0 && h_in < input_height &&
                        w_in >= 0 && w_in < input_width) {
                        
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            int input_idx = n * (in_channels * input_depth * input_height * input_width) +
                                          c_in * (input_depth * input_height * input_width) +
                                          d_in * (input_height * input_width) +
                                          h_in * input_width +
                                          w_in;
                                          
                            int weight_idx = c_out * (in_channels * kernel_depth * kernel_height * kernel_width) +
                                           c_in * (kernel_depth * kernel_height * kernel_width) +
                                           kd * (kernel_height * kernel_width) +
                                           kh * kernel_width +
                                           kw;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    output[out_idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding
) {
    // Set the CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_depth = input_sizes[2];
    int input_height = input_sizes[3];
    int input_width = input_sizes[4];
    
    int out_channels = weight_sizes[0];
    int kernel_depth = weight_sizes[2];
    int kernel_height = weight_sizes[3];
    int kernel_width = weight_sizes[4];
    
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    
    int padding_d = padding[0];
    int padding_h = padding[1];
    int padding_w = padding[2];
    
    int output_padding_d = output_padding[0];
    int output_padding_h = output_padding[1];
    int output_padding_w = output_padding[2];
    
    int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;
    
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        output_padding_d,
        output_padding_h,
        output_padding_w
    );
    
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding
);
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
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.
    Optimized with custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Only groups=1 is supported in this custom implementation")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        
        # Initialize bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
        self.conv_transpose3d_cuda = conv_transpose3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth_in, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        output = self.conv_transpose3d_cuda.conv_transpose3d_cuda(
            x, 
            self.weight, 
            self.stride, 
            self.padding, 
            self.output_padding
        )
        
        if self.bias is not None:
            # Add bias: shape (out_channels,) -> (1, out_channels, 1, 1, 1) for broadcasting
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output