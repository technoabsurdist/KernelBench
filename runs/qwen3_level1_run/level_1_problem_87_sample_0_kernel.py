import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for pointwise convolution (1x1 conv2d)
pointwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void pointwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int out_c = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        float sum = 0.0f;
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            int input_idx = ((b * in_channels + in_c) * height + h) * width + w;
            int weight_idx = out_c * in_channels + in_c;
            sum += input[input_idx] * weight[weight_idx];
        }
        
        if (has_bias) {
            sum += bias[out_c];
        }
        
        output[idx] = sum;
    }
}

torch::Tensor pointwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool has_bias
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    pointwise_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        has_bias
    );
    
    return output;
}
"""

pointwise_conv2d_cpp_source = """
torch::Tensor pointwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool has_bias
);
"""

# Compile the inline CUDA code for pointwise convolution
pointwise_conv2d = load_inline(
    name="pointwise_conv2d",
    cpp_sources=pointwise_conv2d_cpp_source,
    cuda_sources=pointwise_conv2d_source,
    functions=["pointwise_conv2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation with custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = bias
        
        # Initialize weight parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        
        # Initialize bias parameters if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.pointwise_conv2d = pointwise_conv2d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        bias_tensor = self.bias if self.has_bias else torch.empty(0, device=x.device)
        return self.pointwise_conv2d.pointwise_conv2d_cuda(x, self.weight, bias_tensor, self.has_bias)