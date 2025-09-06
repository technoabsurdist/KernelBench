import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D transposed convolution
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
    int in_length,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_pos = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch >= out_channels || out_pos >= (in_length - 1) * stride - 2 * padding + kernel_size + output_padding) 
        return;
        
    float sum = 0.0f;
    
    // For each input channel
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        // Compute the range of kernel positions that contribute to this output position
        int k_min = max(0, (out_pos + padding - (in_length - 1) * stride) / stride);
        int k_max = min(kernel_size - 1, (out_pos + padding) / stride);
        
        for (int k = k_min; k <= k_max; ++k) {
            int in_pos = (out_pos + padding - k * stride);
            if (in_pos >= 0 && in_pos < in_length) {
                int input_idx = batch_idx * (in_channels * in_length) + in_ch * in_length + in_pos;
                int weight_idx = out_ch * (in_channels * kernel_size) + in_ch * kernel_size + (kernel_size - 1 - k);
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    int output_idx = batch_idx * (out_channels * ((in_length - 1) * stride - 2 * padding + kernel_size + output_padding)) + 
                     out_ch * ((in_length - 1) * stride - 2 * padding + kernel_size + output_padding) + out_pos;
    output[output_idx] = sum;
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding
) {
    // Get tensor dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_length = input.size(2);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    // Calculate output dimensions
    auto out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_length}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Set up CUDA launch parameters
    const int threads_per_block = 256;
    const dim3 blocks_per_grid(batch_size, out_channels, (out_length + threads_per_block - 1) / threads_per_block);
    
    // Launch kernel
    at::cuda::CUDAGuard device_guard(input.device());
    conv_transpose1d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_length,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding
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
    int output_padding
);
"""

# Compile the inline CUDA code
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
    Optimized version of Model using custom CUDA kernel for 1D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Only groups=1 is supported in this custom implementation")
        if bias:
            raise NotImplementedError("Bias is not supported in this custom implementation")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return conv_transpose1d.conv_transpose1d_cuda(x, self.weight, self.stride, self.padding, self.output_padding)