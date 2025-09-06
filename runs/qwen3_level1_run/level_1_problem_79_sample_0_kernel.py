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
    int input_length,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_length
) {
    int batch_idx = blockIdx.x;
    int out_ch_idx = blockIdx.y;
    int out_pos = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch_idx >= out_channels || out_pos >= output_length) return;
    
    float sum = 0.0f;
    
    // For each input channel
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        // Compute the range of kernel positions that contribute to this output position
        int k_min = ceilf(fmaxf(0, (float)(out_pos + padding - (input_length - 1) * stride) / dilation));
        int k_max = floorf(fminf(kernel_size - 1, (float)(out_pos + padding) / dilation));
        
        for (int k = k_min; k <= k_max; k++) {
            int in_pos = (out_pos + padding - k * dilation) / stride;
            if ((out_pos + padding - k * dilation) % stride == 0 && in_pos >= 0 && in_pos < input_length) {
                int input_idx = batch_idx * (in_channels * input_length) + in_ch * input_length + in_pos;
                int weight_idx = out_ch_idx * (in_channels * kernel_size) + in_ch * kernel_size + (kernel_size - 1 - k);
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    int output_idx = batch_idx * (out_channels * output_length) + out_ch_idx * output_length + out_pos;
    output[output_idx] = sum;
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    // Get tensor dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    // Calculate output dimensions
    auto output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, output_length}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Set up CUDA launch parameters
    const int threads_per_block = 256;
    const dim3 blocks_per_grid(batch_size, out_channels, (output_length + threads_per_block - 1) / threads_per_block);
    
    // Launch kernel
    at::cuda::CUDAGuard device_guard(input.device());
    conv_transpose1d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_length,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        output_length
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
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.
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
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        
        # Initialize bias parameter if needed
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None
            
        # Register the custom CUDA function
        self.conv_transpose1d_fn = conv_transpose1d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Apply custom CUDA transposed convolution
        output = self.conv_transpose1d_fn.conv_transpose1d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation
        )
        
        # Add bias if needed
        if self.bias and self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1)
            
        return output