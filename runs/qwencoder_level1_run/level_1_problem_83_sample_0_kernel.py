import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise convolution with asymmetric kernel
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int out_height,
    int out_width
) {
    int out_ch = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z;
    int tid = threadIdx.x;
    
    if (out_ch >= in_channels || out_y >= out_height || out_x >= out_width) return;
    
    int batch_idx = 0; // Process one batch element at a time
    float sum = 0.0f;
    
    for (int ky = 0; ky < kernel_h; ky++) {
        for (int kx = 0; kx < kernel_w; kx++) {
            int in_y = out_y * stride_h - pad_h + ky * dilation_h;
            int in_x = out_x * stride_w - pad_w + kx * dilation_w;
            
            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                int input_idx = batch_idx * (in_channels * height * width) + 
                               out_ch * (height * width) + 
                               in_y * width + in_x;
                int weight_idx = out_ch * (kernel_h * kernel_w) + ky * kernel_w + kx;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    int output_idx = batch_idx * (in_channels * out_height * out_width) + 
                    out_ch * (out_height * out_width) + 
                    out_y * out_width + out_x;
    output[output_idx] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    auto out_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    auto out_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    if (batch_size == 0 || in_channels == 0) return output;
    
    dim3 grid(in_channels, out_height, out_width);
    dim3 block(1);
    
    depthwise_conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        out_height,
        out_width
    );
    
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
);
"""

# Compile the inline CUDA code for depthwise convolution
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.
    Optimized with custom CUDA kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Create weight parameter with shape (in_channels, 1, kernel_size, 1)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(in_channels))
        else:
            self.bias_param = None
            
        self.depthwise_conv2d = depthwise_conv2d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Reshape weight to (in_channels, 1, kernel_size, 1) for our kernel
        weight = self.weight.view(self.in_channels, 1, self.kernel_size, 1)
        
        # Call custom CUDA kernel
        output = self.depthwise_conv2d.depthwise_conv2d_cuda(
            x,
            weight,
            self.stride,
            1,  # stride_w = 1 for asymmetric kernel
            self.padding,
            0,  # pad_w = 0 for asymmetric kernel
            self.dilation,
            1   # dilation_w = 1 for asymmetric kernel
        )
        
        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1, 1)
            
        return output