import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % output_width;
    out_idx /= output_width;
    int h = out_idx % output_height;
    out_idx /= output_height;
    int d = out_idx % output_depth;
    out_idx /= output_depth;
    int oc = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    float sum = 0.0f;
    
    // Calculate input position that would contribute to this output position
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate corresponding input position
                int in_d = d + padding - kd;
                int in_h = h + padding - kh;
                int in_w = w + padding - kw;
                
                // Check if within valid input bounds after accounting for stride
                if (in_d % stride == 0 && in_h % stride == 0 && in_w % stride == 0) {
                    in_d /= stride;
                    in_h /= stride;
                    in_w /= stride;
                    
                    if (in_d >= 0 && in_d < input_depth &&
                        in_h >= 0 && in_h < input_height &&
                        in_w >= 0 && in_w < input_width) {
                        
                        // For groups=1, in_channels channels per group
                        for (int ic = 0; ic < in_channels; ic++) {
                            int input_idx = b * (in_channels * input_depth * input_height * input_width) +
                                          ic * (input_depth * input_height * input_width) +
                                          in_d * (input_height * input_width) +
                                          in_h * input_width +
                                          in_w;
                                          
                            int weight_idx = oc * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           ic * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) +
                                           kh * kernel_size +
                                           kw;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    output[out_idx * (out_channels * output_depth * output_height * output_width) +
           oc * (output_depth * output_height * output_width) +
           d * (output_height * output_width) +
           h * output_width +
           w] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding
) {
    // Get input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assuming cubic kernel
    
    // Calculate output dimensions
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch kernel
    const int threads_per_block = 256;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    at::cuda::CUDAGuard device_guard(input.device());
    
    conv_transpose3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding
    );
    
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding
);
"""

# Compile the inline CUDA code
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel using custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Create weight parameter - for groups=1, shape is (out_channels, in_channels, k, k, k)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
        
        # Create bias parameter if needed
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias_param = None
            
        # Store the custom CUDA function
        self.conv_transpose3d_cuda = conv_transpose3d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Call custom CUDA kernel
        output = self.conv_transpose3d_cuda.conv_transpose3d_cuda(
            x, 
            self.weight, 
            self.stride, 
            self.padding, 
            self.output_padding
        )
        
        # Add bias if needed
        if self.bias and self.bias_param is not None:
            # Add bias to the output
            output += self.bias_param.view(1, -1, 1, 1, 1)
            
        return output