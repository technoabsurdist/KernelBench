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
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width,
    bool has_bias
) {
    int out_ch = blockIdx.x;
    int batch = blockIdx.y;
    int out_y = threadIdx.y + blockIdx.z * blockDim.y;
    int out_x = threadIdx.x + blockIdx.z * blockDim.x * gridDim.z;
    
    if (out_ch >= out_channels || batch >= batch_size || out_y >= out_height || out_x >= out_width)
        return;
        
    int kernel_radius = (kernel_size - 1) / 2;
    
    // Depthwise convolution
    float depthwise_result = 0.0f;
    if (out_ch < in_channels) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = batch * in_channels * in_height * in_width +
                                    out_ch * in_height * in_width +
                                    in_y * in_width + in_x;
                    int weight_idx = out_ch * kernel_size * kernel_size +
                                     ky * kernel_size + kx;
                                     
                    depthwise_result += input[input_idx] * depthwise_weight[weight_idx];
                }
            }
        }
    }
    
    // Pointwise convolution
    float pointwise_result = 0.0f;
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        int depthwise_idx = batch * in_channels * out_height * out_width +
                            in_ch * out_height * out_width +
                            out_y * out_width + out_x;
        int weight_idx = out_ch * in_channels + in_ch;
        
        pointwise_result += depthwise_result * pointwise_weight[weight_idx];
    }
    
    if (has_bias) {
        pointwise_result += bias[out_ch];
    }
    
    int output_idx = batch * out_channels * out_height * out_width +
                     out_ch * out_height * out_width +
                     out_y * out_width + out_x;
    output[output_idx] = pointwise_result;
}

torch::Tensor depthwise_separable_conv_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
) {
    // Set the CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto batch_size = input_sizes[0];
    auto in_channels = input_sizes[1];
    auto in_height = input_sizes[2];
    auto in_width = input_sizes[3];
    
    auto kernel_size = static_cast<int>(sqrt(depthwise_weight.sizes()[2] * depthwise_weight.sizes()[3]));
    auto out_channels = pointwise_weight.sizes()[0];
    
    auto out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    bool has_bias = bias.defined() && bias.size(0) > 0;
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size(out_channels, batch_size, (out_height * out_width + 255) / 256);
    
    depthwise_conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        depthwise_weight.data_ptr<float>(),
        pointwise_weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        out_height,
        out_width,
        has_bias
    );
    
    return output;
}
"""

depthwise_separable_conv_cpp_source = """
torch::Tensor depthwise_separable_conv_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
);
"""

# Compile the inline CUDA code for depthwise separable convolution
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
    Performs a depthwise-separable 2D convolution operation with custom CUDA implementation.

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        
        # Initialize depthwise convolution weights
        self.depthwise_weight = nn.Parameter(
            torch.randn(in_channels, 1, kernel_size, kernel_size)
        )
        
        # Initialize pointwise convolution weights
        self.pointwise_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, 1, 1)
        )
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.depthwise_separable_conv = depthwise_separable_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution with custom CUDA implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.depthwise_separable_conv.depthwise_separable_conv_cuda(
            x,
            self.depthwise_weight,
            self.pointwise_weight,
            self.bias if self.has_bias else torch.empty(0, device=x.device),
            self.stride,
            self.padding,
            self.dilation
        )