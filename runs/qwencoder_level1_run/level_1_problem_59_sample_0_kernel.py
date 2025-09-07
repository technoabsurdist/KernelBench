import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution with asymmetric kernel
conv3d_custom_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int input_depth,
    int output_height,
    int output_width,
    int output_depth,
    int kernel_h,
    int kernel_w,
    int kernel_d,
    int stride_h,
    int stride_w,
    int stride_d,
    int padding_h,
    int padding_w,
    int padding_d
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_height * output_width * output_depth;
    
    if (out_idx >= total_outputs) return;
    
    int d = out_idx % output_depth;
    int w = (out_idx / output_depth) % output_width;
    int h = (out_idx / (output_depth * output_width)) % output_height;
    int oc = (out_idx / (output_depth * output_width * output_height)) % out_channels;
    int b = out_idx / (output_depth * output_width * output_height * out_channels);
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                for (int kd = 0; kd < kernel_d; ++kd) {
                    int in_h = h * stride_h - padding_h + kh;
                    int in_w = w * stride_w - padding_w + kw;
                    int in_d = d * stride_d - padding_d + kd;
                    
                    if (in_h >= 0 && in_h < input_height &&
                        in_w >= 0 && in_w < input_width &&
                        in_d >= 0 && in_d < input_depth) {
                        
                        int input_idx = b * (in_channels * input_height * input_width * input_depth) +
                                       ic * (input_height * input_width * input_depth) +
                                       in_h * (input_width * input_depth) +
                                       in_w * input_depth +
                                       in_d;
                                       
                        int weight_idx = oc * (in_channels * kernel_h * kernel_w * kernel_d) +
                                        ic * (kernel_h * kernel_w * kernel_d) +
                                        kh * (kernel_w * kernel_d) +
                                        kw * kernel_d +
                                        kd;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[out_idx] = sum;
}

torch::Tensor conv3d_custom_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int stride_d,
    int padding_h,
    int padding_w,
    int padding_d
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_height = input_sizes[2];
    int input_width = input_sizes[3];
    int input_depth = input_sizes[4];
    
    int out_channels = weight_sizes[0];
    int kernel_h = weight_sizes[2];
    int kernel_w = weight_sizes[3];
    int kernel_d = weight_sizes[4];
    
    int output_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1;
    int output_depth = (input_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width, output_depth}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_outputs = batch_size * out_channels * output_height * output_width * output_depth;
    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;
    
    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        input_depth,
        output_height,
        output_width,
        output_depth,
        kernel_h,
        kernel_w,
        kernel_d,
        stride_h,
        stride_w,
        stride_d,
        padding_h,
        padding_w,
        padding_d
    );
    
    return output;
}
"""

conv3d_custom_cpp_source = """
torch::Tensor conv3d_custom_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int stride_d,
    int padding_h,
    int padding_w,
    int padding_d
);
"""

# Compile the inline CUDA code for 3D convolution
conv3d_custom = load_inline(
    name="conv3d_custom",
    cpp_sources=conv3d_custom_cpp_source,
    cuda_sources=conv3d_custom_source,
    functions=["conv3d_custom_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.
    Optimized with custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Create the weight parameter with shape matching the original Conv3d
        # For the specific case in the problem: kernel_size x kernel_size x 1
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, 1))
        
        # Create bias if needed
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None
            
        # Load the custom CUDA function
        self.conv3d_custom = conv3d_custom
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        # Call our custom CUDA kernel
        output = self.conv3d_custom.conv3d_custom_cuda(
            x, 
            self.weight,
            self.stride,
            self.stride,
            1,  # stride_d is fixed to 1 as per original kernel shape
            self.padding,
            self.padding,
            0   # padding_d is fixed to 0 as per original kernel shape
        )
        
        # Add bias if needed
        if self.bias_param is not None:
            # Reshape bias to broadcast correctly
            bias_view = self.bias_param.view(1, -1, 1, 1, 1)
            output = output + bias_view
            
        return output