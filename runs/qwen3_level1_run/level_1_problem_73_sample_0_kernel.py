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
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int groups
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (out_idx >= total_out_elements) return;
    
    int tmp = out_idx;
    int w_idx = tmp % out_width;
    tmp /= out_width;
    int h_idx = tmp % out_height;
    tmp /= out_height;
    int d_idx = tmp % out_depth;
    tmp /= out_depth;
    int c_out = tmp % out_channels;
    int b_idx = tmp / out_channels;
    
    float sum = 0.0f;
    
    int group_id = c_out * groups / out_channels;
    int in_ch_start = group_id * (in_channels / groups);
    int in_ch_end = (group_id + 1) * (in_channels / groups);
    
    for (int c_in = in_ch_start; c_in < in_ch_end; c_in++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_d = d_idx + padding_d - kd;
                    int in_h = h_idx + padding_h - kh;
                    int in_w = w_idx + padding_w - kw;
                    
                    if (in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                        in_d /= stride_d;
                        in_h /= stride_h;
                        in_w /= stride_w;
                        
                        if (in_d >= 0 && in_d < in_depth &&
                            in_h >= 0 && in_h < in_height &&
                            in_w >= 0 && in_w < in_width) {
                            
                            int input_idx = b_idx * (in_channels * in_depth * in_height * in_width) +
                                          c_in * (in_depth * in_height * in_width) +
                                          in_d * (in_height * in_width) +
                                          in_h * in_width +
                                          in_w;
                                          
                            int weight_idx = c_out * (in_channels * kernel_d * kernel_h * kernel_w) +
                                           c_in * (kernel_d * kernel_h * kernel_w) +
                                           kd * (kernel_h * kernel_w) +
                                           kh * kernel_w +
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
    torch::IntArrayRef padding
) {
    // Set the CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    auto stride_d = stride[0];
    auto stride_h = stride[1];
    auto stride_w = stride[2];
    
    auto padding_d = padding[0];
    auto padding_h = padding[1];
    auto padding_w = padding[2];
    
    auto out_depth = (in_depth - 1) * stride_d - 2 * padding_d + kernel_d;
    auto out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_h;
    auto out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    auto total_out_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    const int block_size = 256;
    const int num_blocks = (total_out_elements + block_size - 1) / block_size;
    
    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        out_depth,
        out_height,
        out_width,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        1  // groups (simplified for this implementation)
    );
    
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding
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
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        
        # Use PyTorch's ConvTranspose3d to initialize weights
        self.conv_transpose3d_ref = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=(kernel_size, kernel_size, kernel_size), 
            stride=stride, padding=padding, 
            groups=groups, bias=bias
        )
        
        # Load custom CUDA module
        self.conv_transpose3d_cuda = conv_transpose3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Use custom CUDA implementation
        output = self.conv_transpose3d_cuda.conv_transpose3d_cuda(
            x, 
            self.conv_transpose3d_ref.weight,
            [self.stride, self.stride, self.stride],
            [self.padding, self.padding, self.padding]
        )
        
        # Add bias if needed
        if self.bias:
            output += self.conv_transpose3d_ref.bias.view(1, -1, 1, 1, 1)
            
        return output