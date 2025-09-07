import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D convolution
conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int dilation,
    bool has_bias
) {
    int batch_idx = blockIdx.x;
    int out_ch_idx = blockIdx.y;
    int out_pos = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch_idx >= out_channels || out_pos >= output_length) {
        return;
    }
    
    float sum = 0.0f;
    
    for (int k = 0; k < kernel_size; ++k) {
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            int input_pos = out_pos * stride + k * dilation;
            if (input_pos < input_length) {
                int input_idx = batch_idx * (in_channels * input_length) + 
                               in_ch * input_length + input_pos;
                int weight_idx = out_ch_idx * (in_channels * kernel_size) + 
                               in_ch * kernel_size + k;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    if (has_bias) {
        sum += bias[out_ch_idx];
    }
    
    int output_idx = batch_idx * (out_channels * output_length) + 
                    out_ch_idx * output_length + out_pos;
    output[output_idx] = sum;
}

torch::Tensor conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation,
    bool has_bias
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_length = (input_length - (kernel_size - 1) * dilation - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_length}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    dim3 grid(batch_size, out_channels, (output_length + 255) / 256);
    dim3 block(256);
    
    conv1d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation,
        has_bias
    );
    
    return output;
}
"""

conv1d_cpp_source = """
torch::Tensor conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation,
    bool has_bias
);
"""

# Compile the inline CUDA code for 1D convolution
conv1d = load_inline(
    name="conv1d",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_source,
    functions=["conv1d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.has_bias = bias
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.conv1d_op = conv1d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return self.conv1d_op.conv1d_cuda(x, self.weight, bias_tensor, self.stride, self.dilation, self.has_bias)