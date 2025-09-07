import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv_transpose1d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_idx = blockIdx.x;
    int out_ch_idx = blockIdx.y;
    int out_pos = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch_idx >= out_channels || out_pos >= output_length) return;
    
    float sum = 0.0f;
    
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_pos = out_pos - k * dilation + 2 * padding;
            if (in_pos % stride == 0) {
                in_pos /= stride;
                if (in_pos >= 0 && in_pos < input_length) {
                    int input_idx = batch_idx * (in_channels * input_length) + 
                                   in_ch * input_length + in_pos;
                    int weight_idx = out_ch_idx * (in_channels * kernel_size) + 
                                    in_ch * kernel_size + (kernel_size - 1 - k);
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    int output_idx = batch_idx * (out_channels * output_length) + 
                     out_ch_idx * output_length + out_pos;
    output[output_idx] = sum;
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    auto output = torch::zeros({batch_size, out_channels, output_length}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int threads_per_block = 256;
    const dim3 blocks(batch_size, out_channels, (output_length + threads_per_block - 1) / threads_per_block);
    
    conv_transpose1d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}
"""

conv_transpose1d_cpp_source = """
torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
);
"""

# Compile the inline CUDA code for transposed 1D convolution
conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None
            
        self.conv_transpose1d = conv_transpose1d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        output = self.conv_transpose1d.conv_transpose1d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        
        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1)
            
        return output