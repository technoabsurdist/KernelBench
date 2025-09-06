import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution with asymmetric kernel (kernel_size x kernel_size x 1)
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
    int kernel_size,
    int stride,
    int padding,
    int dilation
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
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int iy = h * stride - padding + ky * dilation;
                int ix = w * stride - padding + kx * dilation;
                int iz = d; // No convolution along depth dimension
                
                if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
                    int input_idx = b * (in_channels * input_height * input_width * input_depth) +
                                    ic * (input_height * input_width * input_depth) +
                                    iy * (input_width * input_depth) +
                                    ix * input_depth +
                                    iz;
                                    
                    int weight_idx = oc * (in_channels * kernel_size * kernel_size * 1) +
                                     ic * (kernel_size * kernel_size * 1) +
                                     ky * (kernel_size * 1) +
                                     kx * 1 +
                                     0;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    output[out_idx] = sum;
}

torch::Tensor conv3d_custom_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
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
    int kernel_size = weight_sizes[2]; // Assuming square kernel
    
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_depth = input_depth; // No change in depth dimension
    
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
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}
"""

conv3d_custom_cpp_source = """
torch::Tensor conv3d_custom_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
);
"""

# Compile the inline CUDA code for custom 3D convolution
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
    Optimized version of Model using custom CUDA kernel for 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Groups != 1 is not supported in this custom implementation")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Create weight parameter with shape (out_channels, in_channels, kernel_size, kernel_size, 1)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, 1))
        
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None
            
        self.conv3d_custom = conv3d_custom
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        output = self.conv3d_custom.conv3d_custom_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        
        if self.bias and self.bias_param is not None:
            # Add bias to the output
            output += self.bias_param.view(1, -1, 1, 1, 1)
            
        return output