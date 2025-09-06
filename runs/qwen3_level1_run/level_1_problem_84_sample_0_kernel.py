import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int height_out,
    int width_out,
    int kernel_size,
    int stride,
    int padding,
    bool has_bias
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (out_x >= width_out || out_y >= height_out || ch >= out_channels) return;
    
    float sum = 0.0f;
    int half_kernel = kernel_size / 2;
    
    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            int in_y = out_y * stride - padding + ky;
            int in_x = out_x * stride - padding + kx;
            
            if (in_y >= 0 && in_y < height_in && in_x >= 0 && in_x < width_in) {
                int input_idx = ((/*batch*/ 0 * in_channels + ch) * height_in + in_y) * width_in + in_x;
                int weight_idx = (ch * kernel_size + ky) * kernel_size + kx;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    if (has_bias) {
        sum += bias[ch];
    }
    
    int output_idx = ((/*batch*/ 0 * out_channels + ch) * height_out + out_y) * width_out + out_x;
    output[output_idx] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    bool has_bias
) {
    // Assuming input is (N, C, H, W)
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2); // Assuming square kernel
    
    auto height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    auto width_out = (width_in + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    dim3 block(16, 16, 1);
    dim3 grid((width_out + block.x - 1) / block.x,
              (height_out + block.y - 1) / block.y,
              out_channels);
    
    for (int batch = 0; batch < batch_size; batch++) {
        auto input_slice = input[batch];
        auto output_slice = output[batch];
        
        depthwise_conv2d_kernel<<<grid, block>>>(
            input_slice.data_ptr<float>(),
            weight.data_ptr<float>(),
            has_bias ? bias.data_ptr<float>() : nullptr,
            output_slice.data_ptr<float>(),
            1, // batch_size=1 for slice
            in_channels,
            out_channels,
            height_in,
            width_in,
            height_out,
            width_out,
            kernel_size,
            stride,
            padding,
            has_bias
        );
    }
    
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    bool has_bias
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
    Performs a depthwise 2D convolution with asymmetric input and square kernel using custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        
        # For depthwise convolution, out_channels should be equal to in_channels (or multiple)
        # But we'll keep the interface consistent with the original model
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.depthwise_conv2d.depthwise_conv2d_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.empty(0, device=x.device), 
            self.stride, 
            self.padding, 
            self.has_bias
        )