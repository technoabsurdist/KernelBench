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
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_elements) return;
    
    int tmp = out_idx;
    int w_idx = tmp % output_w;
    tmp /= output_w;
    int h_idx = tmp % output_h;
    tmp /= output_h;
    int d_idx = tmp % output_d;
    tmp /= output_d;
    int out_ch = tmp % out_channels;
    int batch = tmp / out_channels;
    
    float sum = 0.0f;
    
    // Compute input coordinates that would contribute to this output position
    for (int kd = 0; kd < kernel_d; kd++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                // Calculate corresponding input position
                int in_d = d_idx + padding_d - kd * dilation_d;
                int in_h = h_idx + padding_h - kh * dilation_h;
                int in_w = w_idx + padding_w - kw * dilation_w;
                
                // Check if the input position is valid after accounting for stride
                if (in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                    in_d /= stride_d;
                    in_h /= stride_h;
                    in_w /= stride_w;
                    
                    if (in_d >= 0 && in_d < input_d && 
                        in_h >= 0 && in_h < input_h && 
                        in_w >= 0 && in_w < input_w) {
                        
                        // Find corresponding input channel
                        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                            int input_idx = batch * (in_channels * input_d * input_h * input_w) +
                                          in_ch * (input_d * input_h * input_w) +
                                          in_d * (input_h * input_w) +
                                          in_h * input_w +
                                          in_w;
                                          
                            int weight_idx = out_ch * (in_channels * kernel_d * kernel_h * kernel_w) +
                                           in_ch * (kernel_d * kernel_h * kernel_w) +
                                           kd * (kernel_h * kernel_w) +
                                           kh * kernel_w +
                                           kw;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    output[out_idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding,
    torch::IntArrayRef dilation
) {
    // Set the CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];
    
    int out_channels = weight_sizes[0];
    int kernel_d = weight_sizes[2];
    int kernel_h = weight_sizes[3];
    int kernel_w = weight_sizes[4];
    
    // Calculate output dimensions
    int output_d = (input_d - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_d - 1) + output_padding[0] + 1;
    int output_h = (input_h - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_h - 1) + output_padding[1] + 1;
    int output_w = (input_w - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_w - 1) + output_padding[2] + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        dilation[0], dilation[1], dilation[2]
    );
    
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding,
    torch::IntArrayRef dilation
);
"""

# Compile the inline CUDA code for 3D transposed convolution
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
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel,
    optimized with custom CUDA kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.bias = bias
        
        # Use PyTorch's ConvTranspose3d to handle weight initialization
        self.conv_transpose3d_ref = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                                       stride=stride, padding=padding, output_padding=output_padding, 
                                                       dilation=dilation, groups=groups, bias=bias)
        
        # Expose the custom CUDA function
        self.conv_transpose3d_cuda = conv_transpose3d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Use the custom CUDA implementation
        output = self.conv_transpose3d_cuda.conv_transpose3d_cuda(
            x, 
            self.conv_transpose3d_ref.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation
        )
        
        # Add bias if needed
        if self.bias and self.conv_transpose3d_ref.bias is not None:
            # Reshape bias to be broadcastable
            bias = self.conv_transpose3d_ref.bias.view(1, -1, 1, 1, 1)
            output = output + bias
            
        return output