import torch
import torch.nn as nn
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
    int pad_d,
    int pad_h,
    int pad_w,
    int groups
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % output_width;
    out_idx /= output_width;
    int h = out_idx % output_height;
    out_idx /= output_height;
    int d = out_idx % output_depth;
    out_idx /= output_depth;
    int c_out = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    int group_id = c_out * groups / out_channels;
    
    float sum = 0.0f;
    
    for (int kd = 0; kd < kernel_depth; kd++) {
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                int in_d = d - kd * stride_d + pad_d;
                int in_h = h - kh * stride_h + pad_h;
                int in_w = w - kw * stride_w + pad_w;
                
                if (in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                    in_d /= stride_d;
                    in_h /= stride_h;
                    in_w /= stride_w;
                    
                    if (in_d >= 0 && in_d < input_depth &&
                        in_h >= 0 && in_h < input_height &&
                        in_w >= 0 && in_w < input_width) {
                        
                        int in_c = (c_out * in_channels / out_channels) + (c_out % (in_channels / groups));
                        
                        int input_idx = b * (in_channels * input_depth * input_height * input_width) +
                                       in_c * (input_depth * input_height * input_width) +
                                       in_d * (input_height * input_width) +
                                       in_h * input_width +
                                       in_w;
                                       
                        int weight_idx = c_out * (in_channels / groups * kernel_depth * kernel_height * kernel_width) +
                                        (in_c - group_id * (in_channels / groups)) * (kernel_depth * kernel_height * kernel_width) +
                                        kd * (kernel_height * kernel_width) +
                                        kh * kernel_width +
                                        kw;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[out_idx * (out_channels * output_depth * output_height * output_width) +
           c_out * (output_depth * output_height * output_width) +
           d * (output_height * output_width) +
           h * output_width +
           w] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding,
    int groups
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
    
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    
    // Calculate output dimensions
    int output_depth = (input_depth - 1) * stride_d - 2 * pad_d + kernel_depth + output_padding[0];
    int output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height + output_padding[1];
    int output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width + output_padding[2];
    
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
        pad_d,
        pad_h,
        pad_w,
        groups
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
    torch::IntArrayRef output_padding,
    int groups
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
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.
    Optimized with custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Use PyTorch's ConvTranspose3d to initialize weights
        self.conv_transpose3d_ref = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self.conv_transpose3d = conv_transpose3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        output = self.conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.conv_transpose3d_ref.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )
        
        if self.bias and self.conv_transpose3d_ref.bias is not None:
            # Add bias if needed
            bias = self.conv_transpose3d_ref.bias.view(1, -1, 1, 1, 1)
            output = output + bias
            
        return output