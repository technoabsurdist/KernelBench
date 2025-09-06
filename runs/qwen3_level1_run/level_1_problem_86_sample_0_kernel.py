import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise-separable convolution
depthwise_separable_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* depthwise_weight,
    const float* pointwise_weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int padding,
    int stride
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch = blockIdx.z;
    
    if (out_ch >= out_channels || out_x >= (width + 2 * padding - kernel_size) / stride + 1 || 
        out_y >= (height + 2 * padding - kernel_size) / stride + 1) return;
        
    int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    float sum = 0.0f;
    
    // Depthwise convolution + Pointwise convolution fused
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        float depthwise_sum = 0.0f;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_x = out_x * stride + kx - padding;
                int in_y = out_y * stride + ky - padding;
                
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    int input_idx = ((0 * in_channels + in_ch) * height + in_y) * width + in_x;
                    int weight_idx = (in_ch * kernel_size + ky) * kernel_size + kx;
                    depthwise_sum += input[input_idx] * depthwise_weight[weight_idx];
                }
            }
        }
        
        // Pointwise convolution (1x1)
        int pointwise_weight_idx = out_ch * in_channels + in_ch;
        sum += depthwise_sum * pointwise_weight[pointwise_weight_idx];
    }
    
    int output_idx = ((0 * out_channels + out_ch) * output_height + out_y) * output_width + out_x;
    output[output_idx] = sum;
}

torch::Tensor depthwise_separable_conv_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    int kernel_size,
    int padding,
    int stride
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = pointwise_weight.size(0);
    
    auto output_height = (height + 2 * padding - kernel_size) / stride + 1;
    auto output_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    dim3 block(16, 16, 1);
    dim3 grid((output_width + block.x - 1) / block.x,
              (output_height + block.y - 1) / block.y,
              out_channels);
    
    depthwise_conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        depthwise_weight.data_ptr<float>(),
        pointwise_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        padding,
        stride
    );
    
    return output;
}
"""

depthwise_separable_conv_cpp_source = """
torch::Tensor depthwise_separable_conv_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    int kernel_size,
    int padding,
    int stride
);
"""

# Compile the inline CUDA code for depthwise-separable convolution
depthwise_separable_conv = load_inline(
    name="depthwise_separable_conv",
    cpp_sources=depthwise_separable_conv_cpp_source,
    cuda_sources=depthwise_separable_conv_source,
    functions=["depthwise_separable_conv_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation with custom CUDA kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if dilation != 1:
            raise NotImplementedError("Dilation not supported in custom CUDA kernel")
        if bias:
            raise NotImplementedError("Bias not supported in custom CUDA kernel")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Depthwise convolution weights (groups = in_channels)
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        # Pointwise convolution weights (1x1 convolution)
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        
        self.depthwise_separable_conv = depthwise_separable_conv
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.depthwise_separable_conv.depthwise_separable_conv_cuda(
            x,
            self.depthwise_weight.squeeze(1),  # Remove the singleton dimension
            self.pointwise_weight.squeeze(2).squeeze(2),  # Remove the singleton dimensions
            self.kernel_size,
            self.padding,
            self.stride
        )